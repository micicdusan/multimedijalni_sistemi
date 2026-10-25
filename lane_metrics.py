import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from thresholding import save_image


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MetricSettings:
    lane_width_m: float = 3.7
    visible_road_length_m: float = 30.0
    vehicle_center_fraction: float = 0.5

    def __post_init__(self):
        values = (self.lane_width_m, self.visible_road_length_m, self.vehicle_center_fraction)
        if not np.isfinite(values).all():
            raise ValueError("Metric settings must be finite.")
        if self.lane_width_m <= 0 or self.visible_road_length_m <= 0:
            raise ValueError("Assumed lane width and visible road length must be positive.")
        if not 0 <= self.vehicle_center_fraction <= 1:
            raise ValueError("Vehicle center fraction must be between 0 and 1.")


def polynomial_in_meters(
    coefficients: np.ndarray, x_m_per_pixel: float, y_m_per_pixel: float
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (3,) or not np.isfinite(coefficients).all():
        raise ValueError("A lane polynomial must have three finite coefficients.")
    if not np.isfinite([x_m_per_pixel, y_m_per_pixel]).all() or min(x_m_per_pixel, y_m_per_pixel) <= 0:
        raise ValueError("Pixel-to-meter scales must be finite and positive.")
    return coefficients * np.array([
        x_m_per_pixel / y_m_per_pixel**2,
        x_m_per_pixel / y_m_per_pixel,
        x_m_per_pixel,
    ])


def radius_of_curvature(coefficients_m: np.ndarray, y_m: float) -> float:
    a, b, _ = coefficients_m
    second_derivative = 2 * float(a)
    if abs(second_derivative) <= 1e-12:
        return float("inf")
    slope = 2 * float(a) * y_m + float(b)
    return float((1 + slope**2)**1.5 / abs(second_derivative))


def vehicle_reference_x(
    inverse_matrix: np.ndarray, image_width: int,
    evaluation_y: float, vehicle_center_fraction: float,
) -> float:
    source_center_x = vehicle_center_fraction * (image_width - 1)
    source_centerline = np.array([1.0, 0.0, -source_center_x])
    bird_centerline = inverse_matrix.T @ source_centerline
    if not np.isfinite(bird_centerline).all() or abs(bird_centerline[0]) <= 1e-12:
        raise ValueError("Cannot intersect the projected vehicle centerline with the evaluation row.")
    return float(-(bird_centerline[1] * evaluation_y + bird_centerline[2]) / bird_centerline[0])


def measure_lane_geometry(
    left_coefficients: np.ndarray, right_coefficients: np.ndarray,
    image_size: tuple[int, int], destination_points: np.ndarray,
    inverse_matrix: np.ndarray, settings: MetricSettings,
) -> dict:
    width, height = image_size
    destination_points = np.asarray(destination_points, dtype=np.float64)
    inverse_matrix = np.asarray(inverse_matrix, dtype=np.float64)
    if min(width, height) < 2:
        raise ValueError("Image dimensions must be at least two pixels.")
    if destination_points.shape != (4, 2) or not np.isfinite(destination_points).all():
        raise ValueError("Destination points must contain four finite (x, y) pairs.")
    if inverse_matrix.shape != (3, 3) or not np.isfinite(inverse_matrix).all():
        raise ValueError("The inverse perspective matrix must be a finite 3x3 matrix.")
    reference_lane_width = float(destination_points[1, 0] - destination_points[0, 0])
    if reference_lane_width <= 0:
        raise ValueError("The destination lane width must be positive.")

    x_m_per_pixel = settings.lane_width_m / reference_lane_width
    y_m_per_pixel = settings.visible_road_length_m / (height - 1)
    left_m = polynomial_in_meters(left_coefficients, x_m_per_pixel, y_m_per_pixel)
    right_m = polynomial_in_meters(right_coefficients, x_m_per_pixel, y_m_per_pixel)
    center_m = (left_m + right_m) / 2
    evaluation_y = float(height - 1)
    evaluation_y_m = evaluation_y * y_m_per_pixel
    left_x = float(np.polyval(left_coefficients, evaluation_y))
    right_x = float(np.polyval(right_coefficients, evaluation_y))
    if right_x <= left_x:
        raise ValueError("Lane boundaries must be ordered left to right at the evaluation row.")
    lane_center_x = (left_x + right_x) / 2
    vehicle_x = vehicle_reference_x(
        inverse_matrix, width, evaluation_y, settings.vehicle_center_fraction,
    )
    offset_m = (vehicle_x - lane_center_x) * x_m_per_pixel
    radii = {
        "left": radius_of_curvature(left_m, evaluation_y_m),
        "right": radius_of_curvature(right_m, evaluation_y_m),
        "center": radius_of_curvature(center_m, evaluation_y_m),
    }
    direction = "centered" if abs(offset_m) < 1e-9 else ("right" if offset_m > 0 else "left")
    return {
        "approximate": True,
        "evaluation_y_pixels": evaluation_y,
        "evaluation_y_m": evaluation_y_m,
        "evaluation_reference": "Nearest row of the visible road ROI.",
        "x_m_per_pixel": x_m_per_pixel,
        "y_m_per_pixel": y_m_per_pixel,
        "reference_lane_width_pixels": reference_lane_width,
        "fitted_lane_width_m": (right_x - left_x) * x_m_per_pixel,
        "left_coefficients_m": left_m.tolist(),
        "right_coefficients_m": right_m.tolist(),
        "center_coefficients_m": center_m.tolist(),
        "radius_m": {name: value if np.isfinite(value) else None for name, value in radii.items()},
        "straight": {name: not np.isfinite(value) for name, value in radii.items()},
        "lane_center_x_pixels": lane_center_x,
        "vehicle_reference_x_pixels": vehicle_x,
        "vehicle_offset_m": offset_m,
        "vehicle_offset_direction": direction,
    }


def format_radius(radius: float | None) -> str:
    return "straight" if radius is None else f"{radius:.0f} m"


def render_metrics(fitted_image: np.ndarray, metrics: dict | None, settings: MetricSettings) -> np.ndarray:
    height, width = fitted_image.shape[:2]
    output = np.full((height + 120, width, 3), 255, dtype=np.uint8)
    output[:height] = fitted_image
    if metrics is None:
        lines = ["Metrics unavailable: upstream fit requires review."]
    else:
        radii = metrics["radius_m"]
        offset = abs(metrics["vehicle_offset_m"])
        direction = metrics["vehicle_offset_direction"]
        position = "centered" if direction == "centered" else f"{offset:.2f} m {direction} of lane center"
        lines = [
            f"Approx. radius: center {format_radius(radii['center'])} | left {format_radius(radii['left'])} | right {format_radius(radii['right'])}",
            f"Vehicle offset estimate: {position}",
            f"Assumed scale: {settings.lane_width_m:g} m lane width; {settings.visible_road_length_m:g} m visible road length",
            "Evaluation reference: nearest visible road row",
        ]
    for index, line in enumerate(lines):
        cv2.putText(
            output, line, (15, height + 23 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate lane curvature and vehicle offset from saved lane fits.")
    parser.add_argument("--fits-dir", type=Path, default=PROJECT_DIR / "outputs/lane_fitting/fits")
    parser.add_argument("--plots-dir", type=Path, default=PROJECT_DIR / "outputs/lane_fitting/fitted")
    parser.add_argument("--transforms-dir", type=Path, default=PROJECT_DIR / "outputs/perspective/transforms")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/lane_metrics")
    parser.add_argument("--lane-width-m", type=float, default=3.7)
    parser.add_argument("--road-length-m", type=float, default=30.0)
    parser.add_argument("--vehicle-center-fraction", type=float, default=0.5)
    args = parser.parse_args()
    settings = MetricSettings(args.lane_width_m, args.road_length_m, args.vehicle_center_fraction)
    paths = sorted(args.fits_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No lane-fit NPZ files found in {args.fits_dir}.")
    image_reports = []
    gallery = [
        "# Lane curvature and vehicle offset estimates", "",
        f"Assumed scale: {settings.lane_width_m:g} m lane width and {settings.visible_road_length_m:g} m visible road length.",
        "The vehicle centerline is assumed to coincide with the configured corrected-image centerline.",
        "Offset is evaluated at the nearest visible road row. Positive means right of the lane center.",
        "All metric results are approximate and depend on these assumptions.", "",
        "| Image | Center radius | Left radius | Right radius | Signed offset (m) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for path in paths:
        with np.load(path, allow_pickle=False) as fit:
            width, height = map(int, fit["image_size"])
            valid = bool(fit["valid"])
            left, right = fit["left_coefficients"], fit["right_coefficients"]
        fitted = cv2.imread(str(args.plots_dir / f"{path.stem}.png"))
        if fitted is None or fitted.shape[:2] != (height + 60, width):
            raise ValueError(f"Missing or incorrectly sized fitted plot for {path.stem}.")
        metrics = None
        if valid:
            transform_path = args.transforms_dir / f"{width}x{height}.npz"
            with np.load(transform_path, allow_pickle=False) as transform:
                if not np.array_equal(transform["image_size"], [width, height]):
                    raise ValueError(f"Perspective transform resolution mismatch for {path.stem}.")
                metrics = measure_lane_geometry(
                    left, right, (width, height), transform["destination_points"],
                    transform["inverse_matrix"], settings,
                )
        plot_file = f"annotated/{path.stem}.png"
        save_image(args.output_dir / plot_file, render_metrics(fitted, metrics, settings))
        image_reports.append({
            "image": f"{path.stem}.png", "image_size": [width, height],
            "status": "estimated" if metrics is not None else "unavailable",
            "plot": plot_file, "metrics": metrics,
        })
        if metrics is None:
            gallery.append(f"| {path.stem} | unavailable | unavailable | unavailable | unavailable |")
            print(f"{path.stem}: metrics unavailable")
        else:
            radii = metrics["radius_m"]
            link = f"[{path.stem}]({plot_file})"
            values = " | ".join(format_radius(radii[side]) for side in ("center", "left", "right"))
            gallery.append(f"| {link} | {values} | {metrics['vehicle_offset_m']:+.3f} |")
            print(f"{path.stem}: center radius {format_radius(radii['center'])}; offset {metrics['vehicle_offset_m']:+.3f} m")

    for item in image_reports:
        gallery.extend(("", f"## {item['image']}", "", f"![Lane metric estimates for {item['image']}]({item['plot']})"))
    report = {
        "settings": asdict(settings),
        "assumptions": [
            "The destination control rectangle spans one lane of the assumed physical width.",
            "The full bird's-eye image height represents the assumed visible road length.",
            "The centered, aligned camera makes the configured corrected-image centerline a vehicle-center proxy.",
        ],
        "offset_sign_convention": "Positive: vehicle reference right of lane center; negative: left.",
        "scale_reference": "Destination rectangle width and full bird's-eye height.",
        "evaluation_reference": "Nearest visible road row, ahead of the vehicle.",
        "straight_radius_encoding": "null radius with straight=true represents an infinite radius.",
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "images": image_reports,
    }
    (args.output_dir / "lane_metrics_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gallery.md").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    print(f"Saved metric estimates for {len(paths)} images.")
    print(f"Gallery: {args.output_dir / 'gallery.md'}")


if __name__ == "__main__":
    main()
