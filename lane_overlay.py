import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from lane_metrics import MetricSettings, format_radius, measure_lane_geometry
from thresholding import save_image
from undistort_images import load_calibration, undistort_image


PROJECT_DIR = Path(__file__).resolve().parent


def create_lane_masks(
    left_coefficients: np.ndarray, right_coefficients: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    if min(width, height) < 2:
        raise ValueError("Image dimensions must be at least two pixels.")
    coefficients = np.asarray([left_coefficients, right_coefficients], dtype=np.float64)
    if coefficients.shape != (2, 3) or not np.isfinite(coefficients).all():
        raise ValueError("Lane boundaries must each have three finite coefficients.")
    plot_y = np.arange(height)
    left_x = np.polyval(coefficients[0], plot_y)
    right_x = np.polyval(coefficients[1], plot_y)
    coordinates = np.column_stack((left_x, right_x))
    if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates) > 1e7):
        raise ValueError("Lane coordinates exceed the supported drawing range.")
    if np.any(right_x <= left_x):
        raise ValueError("Lane boundaries must remain ordered left to right.")
    left_points = np.rint(np.column_stack((left_x, plot_y))).astype(np.int32)
    right_points = np.rint(np.column_stack((right_x, plot_y))).astype(np.int32)
    polygon = np.concatenate((left_points, right_points[::-1]))
    fill = np.zeros((height, width), dtype=np.uint8)
    left = np.zeros_like(fill)
    right = np.zeros_like(fill)
    thickness = max(2, round(width * 0.004))
    cv2.fillPoly(fill, [polygon], 255)
    cv2.polylines(left, [left_points], False, 255, thickness, cv2.LINE_AA)
    cv2.polylines(right, [right_points], False, 255, thickness, cv2.LINE_AA)
    return np.dstack((fill, left, right))


def project_lane(
    corrected: np.ndarray, masks: np.ndarray, inverse_matrix: np.ndarray,
    opacity: float = 0.3,
) -> np.ndarray:
    if corrected.ndim != 3 or corrected.shape[2] != 3 or corrected.dtype != np.uint8:
        raise ValueError("The corrected image must be an unsigned 8-bit BGR image.")
    if masks.shape != corrected.shape or masks.dtype != np.uint8:
        raise ValueError("Lane masks must match the corrected image shape and unsigned 8-bit type.")
    inverse_matrix = np.asarray(inverse_matrix, dtype=np.float64)
    if inverse_matrix.shape != (3, 3) or not np.isfinite(inverse_matrix).all():
        raise ValueError("The inverse perspective matrix must be a finite 3x3 matrix.")
    if np.linalg.matrix_rank(inverse_matrix) != 3:
        raise ValueError("The inverse perspective matrix must be invertible.")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Lane-fill opacity must be between 0 and 1.")
    height, width = corrected.shape[:2]
    projected = cv2.warpPerspective(
        masks, inverse_matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    output = corrected.astype(np.float32)
    for channel, color, strength in (
        (0, (0, 255, 0), opacity),
        (1, (255, 255, 0), 1.0),
        (2, (0, 0, 255), 1.0),
    ):
        alpha = projected[:, :, channel:channel + 1].astype(np.float32) / 255 * strength
        output = output * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.rint(output).clip(0, 255).astype(np.uint8)


def render_readout(image: np.ndarray, metrics: dict | None, settings: MetricSettings) -> np.ndarray:
    output = image.copy()
    if metrics is None:
        lines = ["Lane detection unavailable: upstream fit requires review."]
    else:
        offset = abs(metrics["vehicle_offset_m"])
        direction = metrics["vehicle_offset_direction"]
        position = "centered" if direction == "centered" else f"{offset:.2f} m {direction} of lane center"
        lines = [
            f"Approx. lane radius: {format_radius(metrics['radius_m']['center'])}",
            f"Vehicle offset estimate: {position}",
            f"Assumed scale: {settings.lane_width_m:g} m lane / {settings.visible_road_length_m:g} m road",
            "Offset reference: nearest visible road row",
            "Left boundary: cyan | Right boundary: red",
        ]
    scale = min(image.shape[1] / 1280, 1.0)
    font_scale = 0.65 * scale
    padding = max(6, round(14 * scale))
    line_height = max(12, round(29 * scale))
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for line in lines)
    panel_width = min(image.shape[1] - padding, text_width + 2 * padding)
    panel_height = min(image.shape[0] - padding, line_height * len(lines) + padding)
    panel = output[padding:padding + panel_height, padding:padding + panel_width]
    panel[:] = np.rint(panel.astype(np.float32) * 0.3).astype(np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            output, line, (2 * padding, padding + (index + 1) * line_height),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Project fitted lanes onto corrected road images with metric estimates.")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "test_images")
    parser.add_argument("--fits-dir", type=Path, default=PROJECT_DIR / "outputs/lane_fitting/fits")
    parser.add_argument("--transforms-dir", type=Path, default=PROJECT_DIR / "outputs/perspective/transforms")
    parser.add_argument("--calibration", type=Path, default=PROJECT_DIR / "outputs/calibration/camera_calibration.npz")
    parser.add_argument("--metrics-report", type=Path, default=PROJECT_DIR / "outputs/lane_metrics/lane_metrics_report.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/lane_overlay")
    parser.add_argument("--opacity", type=float, default=0.3)
    args = parser.parse_args()
    if not np.isfinite(args.opacity) or not 0 <= args.opacity <= 1:
        raise ValueError("Lane-fill opacity must be between 0 and 1.")
    settings = MetricSettings(**json.loads(args.metrics_report.read_text(encoding="utf-8"))["settings"])
    camera_matrix, distortion, calibration_size = load_calibration(args.calibration)
    paths = sorted(args.images_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No road JPG images found in {args.images_dir}.")
    reports = []
    gallery = [
        "# Lanes projected onto corrected road images", "",
        "Green: detected lane area. Cyan: left polynomial. Red: right polynomial.",
        "Overlays cover the road ROI represented in the saved bird's-eye view.",
        f"Approximate metrics assume a {settings.lane_width_m:g} m lane and {settings.visible_road_length_m:g} m of visible road.",
        "Offset uses the nearest visible road row and the configured camera-center reference.", "",
    ]
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read road image: {path}")
        corrected = undistort_image(image, camera_matrix, distortion, calibration_size)
        height, width = corrected.shape[:2]
        with np.load(args.fits_dir / f"{path.stem}.npz", allow_pickle=False) as fit:
            if not np.array_equal(fit["image_size"], [width, height]):
                raise ValueError(f"Lane fit resolution mismatch for {path.name}.")
            valid = bool(fit["valid"])
            left, right = fit["left_coefficients"], fit["right_coefficients"]
        metrics = None
        overlay = corrected
        if valid:
            with np.load(args.transforms_dir / f"{width}x{height}.npz", allow_pickle=False) as transform:
                if not np.array_equal(transform["image_size"], [width, height]):
                    raise ValueError(f"Perspective transform resolution mismatch for {path.name}.")
                masks = create_lane_masks(left, right, (width, height))
                overlay = project_lane(corrected, masks, transform["inverse_matrix"], args.opacity)
                metrics = measure_lane_geometry(
                    left, right, (width, height), transform["destination_points"],
                    transform["inverse_matrix"], settings,
                )
        plot_file = f"annotated/{path.stem}.png"
        save_image(args.output_dir / plot_file, render_readout(overlay, metrics, settings))
        reports.append({
            "image": path.name, "image_size": [width, height], "plot": plot_file,
            "status": "projected" if valid else "unavailable", "metrics": metrics,
        })
        gallery.extend(("", f"## {path.name}", "", f"![Projected lane for {path.name}]({plot_file})"))
        if not valid:
            gallery.extend(("", "Lane detection unavailable: the upstream fit requires review."))
        print(f"{path.name}: {'projected' if valid else 'unavailable'}")
    report = {
        "metric_settings": asdict(settings), "fill_opacity": args.opacity,
        "coordinate_space": "Lens-distortion-corrected camera image.",
        "overlay_extent": "Road ROI represented by the full bird's-eye canvas.",
        "colors": {"area": "green", "left_boundary": "cyan", "right_boundary": "red"},
        "metric_settings_source": str(args.metrics_report),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "images": reports,
    }
    (args.output_dir / "lane_overlay_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gallery.md").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    print(f"Saved lane overlays for {len(paths)} images.")
    print(f"Gallery: {args.output_dir / 'gallery.md'}")


if __name__ == "__main__":
    main()
