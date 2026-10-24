import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def load_calibration(
    calibration_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    with np.load(calibration_path, allow_pickle=False) as saved:
        camera_matrix = np.asarray(saved["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(saved["distortion_coefficients"], dtype=np.float64)
        image_size = np.asarray(saved["image_size"])

    if camera_matrix.shape != (3, 3) or distortion.size != 5:
        raise ValueError("Calibration must contain a 3x3 matrix and five distortion coefficients.")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration parameters must be finite.")
    if image_size.shape != (2,) or not np.isfinite(image_size).all():
        raise ValueError("Calibration image size must contain a width and height.")
    if np.any(image_size <= 0) or np.any(image_size != np.floor(image_size)):
        raise ValueError("Calibration image dimensions must be positive integers.")

    return camera_matrix, distortion, (int(image_size[0]), int(image_size[1]))


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    calibration_size: tuple[int, int],
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = np.diag([width / calibration_size[0], height / calibration_size[1], 1.0])
    scaled_matrix = scale @ camera_matrix
    return cv2.undistort(image, scaled_matrix, distortion, None, scaled_matrix)


def save_comparison(
    original: np.ndarray, corrected: np.ndarray, output_path: Path, image_name: str
) -> None:
    height, width = original.shape[:2]
    comparison = np.full((height + 60, width * 2, 3), 255, dtype=np.uint8)
    comparison[60:, :width] = original
    comparison[60:, width:] = corrected
    labels = ((f"Original: {image_name}", 20), ("Distortion corrected", width + 20))
    for label, x in labels:
        cv2.putText(
            comparison, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 2, cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Could not save {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct road-image lens distortion and save original/corrected comparisons."
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=PROJECT_DIR / "outputs/calibration/camera_calibration.npz",
    )
    parser.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "test_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/undistortion")
    args = parser.parse_args()

    camera_matrix, distortion, calibration_size = load_calibration(args.calibration)
    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No road JPG images found in {args.images_dir}.")

    scaled_count = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read road image: {image_path}")
        corrected = undistort_image(image, camera_matrix, distortion, calibration_size)
        comparison_path = args.output_dir / "comparisons" / f"{image_path.stem}_comparison.jpg"
        save_comparison(image, corrected, comparison_path, image_path.name)
        image_size = (image.shape[1], image.shape[0])
        scaled_count += image_size != calibration_size
        print(f"Corrected {image_path.name}: {image_size[0]}x{image_size[1]}")

    print(f"Calibration resolution: {calibration_size[0]}x{calibration_size[1]}")
    print(f"Used a scaled camera matrix for {scaled_count} of {len(image_paths)} images.")
    print(f"Saved {len(image_paths)} comparisons to {args.output_dir / 'comparisons'}")


if __name__ == "__main__":
    main()
