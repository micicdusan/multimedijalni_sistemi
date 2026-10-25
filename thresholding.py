import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from undistort_images import load_calibration, undistort_image


PROJECT_DIR = Path(__file__).resolve().parent
METHODS = ("color", "gradient", "or", "and")
METHOD_LABELS = {
    "color": "Color only",
    "gradient": "Gradient only",
    "or": "Color OR gradient",
    "and": "Color AND gradient",
}
PARAMETER_CHECK_IMAGES = ("straight_lines1.jpg", "test2.jpg", "solidYellowLeft.jpg")


@dataclass(frozen=True)
class ThresholdSettings:
    blur_kernel: int = 3
    white_lightness_min: int = 200
    white_saturation_max: int = 100
    yellow_hue_min: int = 15
    yellow_hue_max: int = 40
    yellow_saturation_min: int = 80
    yellow_value_min: int = 80
    sobel_kernel: int = 3
    gradient_min: int = 30
    gradient_max: int = 255


def color_threshold(image: np.ndarray, settings: ThresholdSettings) -> np.ndarray:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hls, (0, settings.white_lightness_min, 0),
        (179, 255, settings.white_saturation_max),
    )
    yellow = cv2.inRange(
        hsv,
        (settings.yellow_hue_min, settings.yellow_saturation_min, settings.yellow_value_min),
        (settings.yellow_hue_max, 255, 255),
    )
    return ((white != 0) | (yellow != 0)).astype(np.uint8)


def gradient_threshold(image: np.ndarray, settings: ThresholdSettings) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=settings.sobel_kernel)
    absolute = np.abs(sobel_x)
    maximum = float(absolute.max())
    if maximum == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    scaled = np.rint(255 * absolute / maximum).astype(np.uint8)
    return ((scaled >= settings.gradient_min) & (scaled <= settings.gradient_max)).astype(np.uint8)


def or_threshold(image: np.ndarray, settings: ThresholdSettings) -> np.ndarray:
    filtered = cv2.GaussianBlur(image, (settings.blur_kernel, settings.blur_kernel), 0)
    color = color_threshold(filtered, settings)
    gradient = gradient_threshold(filtered, settings)
    return color | gradient


def create_masks(
    image: np.ndarray, settings: ThresholdSettings
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    start = perf_counter()
    filtered = cv2.GaussianBlur(image, (settings.blur_kernel, settings.blur_kernel), 0)
    blur_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    color = color_threshold(filtered, settings)
    color_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    gradient = gradient_threshold(filtered, settings)
    gradient_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    union = color | gradient
    or_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    intersection = color & gradient
    and_ms = (perf_counter() - start) * 1000

    masks = {"color": color, "gradient": gradient, "or": union, "and": intersection}
    timings = {
        "color": blur_ms + color_ms,
        "gradient": blur_ms + gradient_ms,
        "or": blur_ms + color_ms + gradient_ms + or_ms,
        "and": blur_ms + color_ms + gradient_ms + and_ms,
    }
    return masks, timings


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")


def save_comparison(
    image: np.ndarray, masks: dict[str, np.ndarray],
    image_name: str, output_path: Path, settings: ThresholdSettings,
) -> None:
    height, width = image.shape[:2]
    header = 60
    tile_height = height + header
    panel = np.full((tile_height * 2, width * 3, 3), 255, dtype=np.uint8)
    tiles = (
        (0, 0, f"Undistorted: {image_name}", image),
        (0, 1, METHOD_LABELS["color"], masks["color"]),
        (0, 2, METHOD_LABELS["gradient"], masks["gradient"]),
        (1, 1, METHOD_LABELS["or"], masks["or"]),
        (1, 2, METHOD_LABELS["and"], masks["and"]),
    )
    for row, column, label, pixels in tiles:
        x = column * width
        y = row * tile_height
        if pixels.ndim == 2:
            pixels = cv2.cvtColor(pixels * 255, cv2.COLOR_GRAY2BGR)
        panel[y + header : y + tile_height, x : x + width] = pixels
        cv2.putText(
            panel, label, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 0, 0), 2, cv2.LINE_AA,
        )

    lines = [
        "Same input and thresholds; no ROI",
        f"Image size: {width} x {height}",
        f"Common Gaussian blur: {settings.blur_kernel} x {settings.blur_kernel}",
        "Active pixels (% of full image):",
    ]
    lines.extend(f"{METHOD_LABELS[name]}: {100 * masks[name].mean():.2f}%" for name in METHODS)
    lines.extend(("Active-pixel count is not accuracy.", "No method selected yet."))
    for index, line in enumerate(lines):
        cv2.putText(
            panel, line, (20, tile_height + 50 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA,
        )
    save_image(output_path, panel)


def save_gallery(output_dir: Path, image_reports: list[dict]) -> None:
    lines = [
        "# Thresholding experiment",
        "",
        "All four methods use the same distortion-corrected input and fixed settings.",
        "No ROI is applied. Review lane coverage and unwanted detections before choosing a method.",
        "Active-pixel percentages measure mask density, rather than accuracy.",
        "",
        "Panel layout: corrected image / color / gradient; experiment details / OR / AND.",
        "Open a PNG at its original size to inspect thin gradient edges.",
        "",
        "| Image | Parameter check or evaluation | Color % | Gradient % | OR % | AND % |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in image_reports:
        name = report["image"]
        values = [f"{100 * report['methods'][method]['active_fraction']:.2f}" for method in METHODS]
        link = f"[{name}](comparisons/{Path(name).stem}_comparison.png)"
        lines.append(f"| {link} | {report['group']} | " + " | ".join(values) + " |")
    for report in image_reports:
        name = report["image"]
        panel = f"comparisons/{Path(name).stem}_comparison.png"
        lines.extend(("", f"## {name}", "", f"![{name}: four thresholding methods]({panel})"))
    (output_dir / "comparison_gallery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare color, gradient, OR, and AND lane masks.")
    parser.add_argument(
        "--calibration", type=Path,
        default=PROJECT_DIR / "outputs/calibration/camera_calibration.npz",
    )
    parser.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "test_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/thresholding")
    args = parser.parse_args()

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No road JPG images found in {args.images_dir}.")
    matrix, distortion, calibration_size = load_calibration(args.calibration)
    settings = ThresholdSettings()
    image_reports = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read road image: {image_path}")
        corrected = undistort_image(image, matrix, distortion, calibration_size)
        masks, timings = create_masks(corrected, settings)
        methods = {}
        for method, mask in masks.items():
            mask_path = args.output_dir / "masks" / method / f"{image_path.stem}.png"
            save_image(mask_path, mask * 255)
            methods[method] = {
                "active_pixels": int(np.count_nonzero(mask)),
                "active_fraction": float(mask.mean()),
                "thresholding_ms": timings[method],
            }
        comparison_path = args.output_dir / "comparisons" / f"{image_path.stem}_comparison.png"
        save_comparison(corrected, masks, image_path.name, comparison_path, settings)
        image_reports.append({
            "image": image_path.name,
            "group": "parameter check" if image_path.name in PARAMETER_CHECK_IMAGES else "evaluation",
            "image_size": [image.shape[1], image.shape[0]],
            "methods": methods,
        })
        print(f"Compared four masks for {image_path.name}")

    report = {
        "settings": asdict(settings),
        "calibration_file": str(args.calibration),
        "calibration_size": list(calibration_size),
        "roi_applied": False,
        "selected_method": None,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "timing_scope": "Single-pass thresholding including shared blur; excludes undistortion and file I/O.",
        "images": image_reports,
    }
    (args.output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    save_gallery(args.output_dir, image_reports)
    print(f"Saved {len(image_paths)} comparison panels and {len(image_paths) * 4} binary masks.")
    print(f"Gallery: {args.output_dir / 'comparison_gallery.md'}")


if __name__ == "__main__":
    main()
