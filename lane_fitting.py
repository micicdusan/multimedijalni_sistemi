import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from thresholding import save_image


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LaneSettings:
    windows: int = 9
    margin_fraction: float = 0.08
    minimum_window_pixel_fraction: float = 0.0002
    minimum_pixel_fraction: float = 0.00015
    minimum_row_fraction: float = 0.1
    minimum_y_span_fraction: float = 0.35
    refinement_iterations: int = 3
    outlier_sigma: float = 3.0
    minimum_residual_fraction: float = 0.01
    minimum_lane_width_fraction: float = 0.25
    maximum_lane_width_fraction: float = 0.8


@dataclass
class LaneBoundary:
    coefficients: np.ndarray
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    candidate_count: int
    rms_error_pixels: float


@dataclass
class LaneDetection:
    left: LaneBoundary | None
    right: LaneBoundary | None
    issues: list[str]


def find_lane_pixels(
    binary: np.ndarray, settings: LaneSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = binary.shape
    histogram = np.count_nonzero(binary[height // 2 :], axis=0)
    midpoint = width // 2
    left_current = int(np.argmax(histogram[:midpoint])) if histogram[:midpoint].any() else None
    right_current = midpoint + int(np.argmax(histogram[midpoint:])) if histogram[midpoint:].any() else None
    margin = max(2, int(round(width * settings.margin_fraction)))
    minimum_pixels = max(10, int(round(binary.size * settings.minimum_window_pixel_fraction)))
    nonzero_y, nonzero_x = np.nonzero(binary)
    edges = np.linspace(height, 0, settings.windows + 1, dtype=int)
    left_indices = []
    right_indices = []

    for bottom, top in zip(edges[:-1], edges[1:]):
        in_band = (nonzero_y >= top) & (nonzero_y < bottom)
        if left_current is not None:
            selected = np.flatnonzero(in_band & (np.abs(nonzero_x - left_current) <= margin))
            if selected.size >= minimum_pixels:
                left_indices.extend(selected)
                left_current = int(np.median(nonzero_x[selected]))
        if right_current is not None:
            selected = np.flatnonzero(in_band & (np.abs(nonzero_x - right_current) <= margin))
            if selected.size >= minimum_pixels:
                right_indices.extend(selected)
                right_current = int(np.median(nonzero_x[selected]))

    left_indices = np.asarray(left_indices, dtype=int)
    right_indices = np.asarray(right_indices, dtype=int)
    return (
        nonzero_x[left_indices], nonzero_y[left_indices],
        nonzero_x[right_indices], nonzero_y[right_indices],
    )


def fit_boundary(
    pixel_x: np.ndarray, pixel_y: np.ndarray,
    image_size: tuple[int, int], settings: LaneSettings,
) -> LaneBoundary | None:
    width, height = image_size
    minimum_pixels = max(30, int(round(width * height * settings.minimum_pixel_fraction)))
    minimum_rows = max(3, int(round(height * settings.minimum_row_fraction)))

    def sufficient_support(keep: np.ndarray) -> bool:
        rows = np.unique(pixel_y[keep])
        return (
            int(np.count_nonzero(keep)) >= minimum_pixels
            and rows.size >= minimum_rows
            and float(np.ptp(rows)) >= settings.minimum_y_span_fraction * (height - 1)
        )

    keep = np.ones(pixel_x.size, dtype=bool)
    if not sufficient_support(keep):
        return None

    for _ in range(settings.refinement_iterations):
        counts = np.bincount(pixel_y[keep], minlength=height)
        weights = 1 / np.sqrt(counts[pixel_y[keep]])
        coefficients = np.polyfit(pixel_y[keep], pixel_x[keep], 2, w=weights)
        residuals = pixel_x - np.polyval(coefficients, pixel_y)
        center = float(np.median(residuals[keep]))
        deviation = float(np.median(np.abs(residuals[keep] - center)))
        limit = max(width * settings.minimum_residual_fraction, settings.outlier_sigma * 1.4826 * deviation)
        refined = np.abs(residuals - center) <= limit
        if np.array_equal(refined, keep) or not sufficient_support(refined):
            break
        keep = refined

    counts = np.bincount(pixel_y[keep], minlength=height)
    weights = 1 / np.sqrt(counts[pixel_y[keep]])
    coefficients = np.polyfit(pixel_y[keep], pixel_x[keep], 2, w=weights)
    residuals = pixel_x[keep] - np.polyval(coefficients, pixel_y[keep])
    return LaneBoundary(
        coefficients, pixel_x[keep], pixel_y[keep], int(pixel_x.size),
        float(np.sqrt(np.mean(residuals**2))),
    )


def detect_lane_boundaries(binary: np.ndarray, settings: LaneSettings) -> LaneDetection:
    if binary.ndim != 2 or min(binary.shape) < 2:
        raise ValueError("A lane mask must be a two-dimensional image of at least 2x2 pixels.")
    if settings.windows < 1 or not 0 < settings.margin_fraction < 0.5:
        raise ValueError("Use at least one search window and a margin fraction between 0 and 0.5.")
    height, width = binary.shape
    left_x, left_y, right_x, right_y = find_lane_pixels(binary, settings)
    left = fit_boundary(left_x, left_y, (width, height), settings)
    right = fit_boundary(right_x, right_y, (width, height), settings)
    issues = []
    if left is None:
        issues.append("Insufficient left-boundary pixel support.")
    if right is None:
        issues.append("Insufficient right-boundary pixel support.")
    if left is not None and right is not None:
        plot_y = np.linspace(0, height - 1, 50)
        lane_width = np.polyval(right.coefficients, plot_y) - np.polyval(left.coefficients, plot_y)
        if np.any(lane_width <= 0):
            issues.append("Fitted lane boundaries cross.")
        elif np.any(lane_width < width * settings.minimum_lane_width_fraction):
            issues.append("Fitted lane is narrower than the configured plausible range.")
        elif np.any(lane_width > width * settings.maximum_lane_width_fraction):
            issues.append("Fitted lane is wider than the configured plausible range.")
    return LaneDetection(left, right, issues)


def render_lane_fit(binary: np.ndarray, detection: LaneDetection, image_name: str) -> np.ndarray:
    height, width = binary.shape
    view = cv2.cvtColor((binary != 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    plot_y = np.arange(height)
    for boundary, color in ((detection.left, (0, 255, 0)), (detection.right, (0, 0, 255))):
        if boundary is None:
            continue
        plot_x = np.polyval(boundary.coefficients, plot_y)
        visible = np.flatnonzero((plot_x >= 0) & (plot_x <= width - 1))
        for indices in np.split(visible, np.flatnonzero(np.diff(visible) > 1) + 1):
            if indices.size < 2:
                continue
            points = np.rint(np.column_stack((plot_x[indices], plot_y[indices]))).astype(np.int32)
            cv2.polylines(view, [points], False, color, max(2, int(round(width * 0.004))), cv2.LINE_AA)

    output = np.full((height + 60, width, 3), 255, dtype=np.uint8)
    output[60:] = view
    title = image_name if not detection.issues else f"{image_name} - review fit"
    cv2.putText(output, title, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "Left polynomial", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 150, 0), 1, cv2.LINE_AA)
    cv2.putText(output, "Right polynomial", (240, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    return output


def boundary_report(boundary: LaneBoundary | None) -> dict | None:
    if boundary is None:
        return None
    return {
        "coefficients": boundary.coefficients.tolist(),
        "candidate_pixels": boundary.candidate_count,
        "inlier_pixels": int(boundary.pixel_x.size),
        "observed_y_range": [int(boundary.pixel_y.min()), int(boundary.pixel_y.max())],
        "observed_rows": int(np.unique(boundary.pixel_y).size),
        "rms_error_pixels": boundary.rms_error_pixels,
    }


def save_fit(path: Path, detection: LaneDetection, image_size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"image_size": np.array(image_size), "valid": np.array(not detection.issues)}
    for name, boundary in (("left", detection.left), ("right", detection.right)):
        arrays[f"{name}_coefficients"] = np.array([]) if boundary is None else boundary.coefficients
        arrays[f"{name}_pixel_x"] = np.array([], dtype=int) if boundary is None else boundary.pixel_x
        arrays[f"{name}_pixel_y"] = np.array([], dtype=int) if boundary is None else boundary.pixel_y
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect lane pixels and fit bird's-eye lane polynomials.")
    parser.add_argument("--masks-dir", type=Path, default=PROJECT_DIR / "outputs/perspective/binary")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/lane_fitting")
    args = parser.parse_args()
    paths = sorted(args.masks_dir.glob("*.png"))
    if not paths:
        raise ValueError(f"No bird's-eye PNG masks found in {args.masks_dir}.")
    settings = LaneSettings()
    reports = []
    gallery = [
        "# Lane pixels and fitted polynomials", "",
        "Bird's-eye OR masks with the left boundary in green and the right boundary in red.",
        "Curves bridge dashed-line gaps. Review flags identify insufficient support or implausible geometry.", "",
    ]

    for path in paths:
        binary = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise ValueError(f"Cannot read bird's-eye mask: {path}")
        binary = (binary != 0).astype(np.uint8)
        detection = detect_lane_boundaries(binary, settings)
        image_size = (binary.shape[1], binary.shape[0])
        plot_file = f"fitted/{path.stem}.png"
        fit_file = f"fits/{path.stem}.npz"
        save_image(args.output_dir / plot_file, render_lane_fit(binary, detection, path.name))
        save_fit(args.output_dir / fit_file, detection, image_size)
        reports.append({
            "image": path.name, "image_size": list(image_size),
            "plot": plot_file, "fit": fit_file,
            "status": "ok" if not detection.issues else "review",
            "issues": detection.issues,
            "left": boundary_report(detection.left), "right": boundary_report(detection.right),
        })
        gallery.extend(("", f"## {path.name}", "", f"![Fitted lane boundaries for {path.name}]({plot_file})"))
        if detection.issues:
            gallery.extend(("", "Review: " + " ".join(detection.issues)))
        print(f"{path.name}: {'ok' if not detection.issues else 'review - ' + ' '.join(detection.issues)}")

    report = {
        "settings": asdict(settings),
        "polynomial": "x = a*y**2 + b*y + c",
        "coefficient_order": ["a", "b", "c"],
        "coordinate_units": "bird's-eye pixels",
        "input_directory": str(args.masks_dir),
        "images": reports,
    }
    (args.output_dir / "lane_fitting_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "gallery.md").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    print(f"Saved {len(paths)} bird's-eye images with fitted polynomials.")
    print(f"Gallery: {args.output_dir / 'gallery.md'}")


if __name__ == "__main__":
    main()
