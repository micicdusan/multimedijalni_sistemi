import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from thresholding import ThresholdSettings, or_threshold, save_image
from undistort_images import load_calibration, undistort_image


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PerspectiveSettings:
    reference_size: tuple[int, int] = (1280, 720)
    source_points: tuple[tuple[float, float], ...] = (
        (580, 460), (705, 460), (1020, 660), (295, 660),
    )
    destination_margin: float = 0.25


@dataclass
class PerspectiveTransform:
    image_size: tuple[int, int]
    source_points: np.ndarray
    destination_points: np.ndarray
    matrix: np.ndarray
    inverse_matrix: np.ndarray


def create_perspective_transform(
    image_size: tuple[int, int], settings: PerspectiveSettings
) -> PerspectiveTransform:
    width, height = image_size
    if min(width, height) < 2 or min(settings.reference_size) < 2:
        raise ValueError("Image and reference dimensions must be at least 2 pixels.")
    if not 0 <= settings.destination_margin < 0.5:
        raise ValueError("Destination margin must be at least 0 and less than 0.5.")
    source = np.asarray(settings.source_points, dtype=np.float32)
    if source.shape != (4, 2) or not np.isfinite(source).all():
        raise ValueError("Source points must contain four finite (x, y) pairs.")
    if np.any(source < 0) or np.any(source >= np.asarray(settings.reference_size)):
        raise ValueError("Source points must be inside the reference image.")
    if not cv2.isContourConvex(source.reshape(4, 1, 2)):
        raise ValueError("Source points must form a nondegenerate convex quadrilateral.")

    source = source * np.float32([
        width / settings.reference_size[0], height / settings.reference_size[1],
    ])
    margin = settings.destination_margin
    destination = np.float32([
        [margin, 0], [1 - margin, 0], [1 - margin, 1], [margin, 1],
    ]) * np.float32([width - 1, height - 1])
    matrix = cv2.getPerspectiveTransform(source, destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, source)
    return PerspectiveTransform(image_size, source, destination, matrix, inverse_matrix)


def warp_binary(binary: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    return cv2.warpPerspective(
        binary, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def get_source_roi(transform: PerspectiveTransform) -> np.ndarray:
    width, height = transform.image_size
    image_bounds = np.float32([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
    ]).reshape(-1, 1, 2)
    footprint = cv2.perspectiveTransform(image_bounds, transform.inverse_matrix)
    area, roi = cv2.intersectConvexConvex(footprint, image_bounds)
    if roi is None or area <= 0:
        raise ValueError("The transformed view has no visible source ROI.")
    return roi.reshape(-1, 2)


def draw_region_of_interest(image: np.ndarray, transform: PerspectiveTransform) -> np.ndarray:
    annotated = image.copy()
    points = np.rint(get_source_roi(transform)).astype(np.int32)
    cv2.polylines(annotated, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
    return annotated


def save_comparison(
    corrected: np.ndarray, warped: np.ndarray, transform: PerspectiveTransform,
    image_name: str, output_path: Path,
) -> None:
    height, width = corrected.shape[:2]
    panel = np.full((height + 60, width * 2, 3), 255, dtype=np.uint8)
    panel[60:, :width] = draw_region_of_interest(corrected, transform)
    panel[60:, width:] = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2BGR)
    labels = ((f"ROI: {image_name}", 20), ("Bird's-eye OR binary mask", width + 20))
    for label, x in labels:
        cv2.putText(
            panel, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 0), 2, cv2.LINE_AA,
        )
    save_image(output_path, panel)


def save_transform(transform: PerspectiveTransform, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        image_size=np.array(transform.image_size),
        source_points=transform.source_points,
        destination_points=transform.destination_points,
        roi_points=get_source_roi(transform),
        matrix=transform.matrix,
        inverse_matrix=transform.inverse_matrix,
    )


def save_gallery(output_dir: Path, image_reports: list[dict]) -> None:
    lines = [
        "# Perspective transformation",
        "",
        "Left: distortion-corrected image with the visible source ROI outlined in green.",
        "Right: bird's-eye binary mask using the selected OR thresholding method.",
        "The ROI is the inverse projection of the output borders, clipped to the source image.",
        "",
    ]
    for report in image_reports:
        name = report["image"]
        lines.extend(("", f"## {name}", "", f"![Perspective transform of {name}]({report['comparison']})"))
    (output_dir / "comparison_gallery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform OR lane masks to a bird's-eye view.")
    parser.add_argument(
        "--calibration", type=Path,
        default=PROJECT_DIR / "outputs/calibration/camera_calibration.npz",
    )
    parser.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "test_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/perspective")
    args = parser.parse_args()

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No road JPG images found in {args.images_dir}.")
    matrix, distortion, calibration_size = load_calibration(args.calibration)
    settings = PerspectiveSettings()
    threshold_settings = ThresholdSettings()
    transforms = {}
    image_reports = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read road image: {image_path}")
        corrected = undistort_image(image, matrix, distortion, calibration_size)
        image_size = (image.shape[1], image.shape[0])
        transform_file = f"transforms/{image_size[0]}x{image_size[1]}.npz"
        if image_size not in transforms:
            transforms[image_size] = create_perspective_transform(image_size, settings)
            save_transform(transforms[image_size], args.output_dir / transform_file)
        transform = transforms[image_size]
        binary = or_threshold(corrected, threshold_settings)
        warped = warp_binary(binary, transform.matrix)
        binary_file = f"binary/{image_path.stem}.png"
        comparison_file = f"comparisons/{image_path.stem}_comparison.png"
        save_image(args.output_dir / binary_file, warped * 255)
        save_comparison(
            corrected, warped, transform, image_path.name, args.output_dir / comparison_file,
        )
        image_reports.append({
            "image": image_path.name,
            "image_size": list(image_size),
            "transform": transform_file,
            "binary_mask": binary_file,
            "comparison": comparison_file,
            "active_pixels": int(np.count_nonzero(warped)),
        })
        print(f"Transformed {image_path.name}: {image_size[0]}x{image_size[1]}")

    report = {
        "threshold_method": "or",
        "threshold_settings": asdict(threshold_settings),
        "perspective_settings": asdict(settings),
        "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "interpolation": "INTER_NEAREST",
        "roi_definition": "Inverse projection of full output bounds, clipped to the source image.",
        "calibration_file": str(args.calibration),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "images": image_reports,
    }
    (args.output_dir / "perspective_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    save_gallery(args.output_dir, image_reports)
    print(f"Saved {len(image_paths)} two-view comparisons and bird's-eye binary masks.")
    print(f"Gallery: {args.output_dir / 'comparison_gallery.md'}")


if __name__ == "__main__":
    main()
