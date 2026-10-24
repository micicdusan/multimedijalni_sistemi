import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]
    board_size: tuple[int, int]
    rms_reprojection_error: float
    per_image_rms_pixels: dict[str, float]
    skipped_images: dict[str, str]


def calibrate_camera(
    image_dir: Path, board_size: tuple[int, int] = (9, 6)
) -> CalibrationResult:
    if min(board_size) < 2:
        raise ValueError("The chessboard must have at least 2 internal corners per axis.")
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No calibration JPG images found in {image_dir}.")

    object_grid = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_grid[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)

    object_points = []
    image_points = []
    used_images = []
    skipped_images = {}
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read calibration image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            skipped_images[image_path.name] = (
                f"Image size {current_size} differs from calibration size {image_size}."
            )
            continue

        found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if not found:
            skipped_images[image_path.name] = f"Internal corner grid {board_size} not found."
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_grid.copy())
        image_points.append(corners)
        used_images.append(image_path.name)

    if len(used_images) < 3:
        raise ValueError(
            f"Only {len(used_images)} usable chessboard images; need at least 3. "
            "Check the internal corner count and supply multiple board poses."
        )

    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    per_image_rms = {}
    for name, points_3d, points_2d, rotation, translation in zip(
        used_images, object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(points_3d, rotation, translation, matrix, distortion)
        residuals = points_2d.reshape(-1, 2) - projected.reshape(-1, 2)
        per_image_rms[name] = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    return CalibrationResult(
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        board_size=board_size,
        rms_reprojection_error=float(rms),
        per_image_rms_pixels=per_image_rms,
        skipped_images=skipped_images,
    )


def save_calibration(result: CalibrationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "camera_calibration.npz",
        camera_matrix=result.camera_matrix,
        distortion_coefficients=result.distortion_coefficients,
        image_size=np.array(result.image_size),
        board_size=np.array(result.board_size),
    )
    report = {
        "image_size": list(result.image_size),
        "board_size": list(result.board_size),
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": result.distortion_coefficients.ravel().tolist(),
        "distortion_coefficient_order": ["k1", "k2", "p1", "p2", "k3"],
        "rms_reprojection_error_pixels": result.rms_reprojection_error,
        "used_images": list(result.per_image_rms_pixels),
        "skipped_images": result.skipped_images,
        "per_image_rms_pixels": result.per_image_rms_pixels,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def save_comparisons(
    result: CalibrationResult, image_dir: Path, output_dir: Path
) -> None:
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for image_name in result.per_image_rms_pixels:
        image_path = image_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read calibration image: {image_path}")
        if (image.shape[1], image.shape[0]) != result.image_size:
            raise ValueError(f"{image_name} resolution must match the calibration resolution.")

        corrected = cv2.undistort(
            image, result.camera_matrix, result.distortion_coefficients,
            None, result.camera_matrix,
        )
        height, width = image.shape[:2]
        comparison = np.full((height + 60, width * 2, 3), 255, dtype=np.uint8)
        comparison[60:, :width] = image
        comparison[60:, width:] = corrected
        labels = ((f"Original: {image_name}", 20), ("Distortion corrected", width + 20))
        for label, x in labels:
            cv2.putText(
                comparison, label, (x, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 2, cv2.LINE_AA,
            )

        comparison_path = comparison_dir / f"{image_path.stem}_comparison.jpg"
        if not cv2.imwrite(str(comparison_path), comparison):
            raise OSError(f"Could not save {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the camera and compare all accepted chessboard photos."
    )
    parser.add_argument("--images-dir", type=Path, default=PROJECT_DIR / "camera_cal")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/calibration")
    parser.add_argument("--columns", type=int, default=9, help="Internal corner columns")
    parser.add_argument("--rows", type=int, default=6, help="Internal corner rows")
    args = parser.parse_args()

    result = calibrate_camera(args.images_dir, (args.columns, args.rows))
    save_comparisons(result, args.images_dir, args.output_dir)
    save_calibration(result, args.output_dir)
    print(f"Used {len(result.per_image_rms_pixels)} images; skipped {len(result.skipped_images)}.")
    for name, reason in result.skipped_images.items():
        print(f"Skipped {name}: {reason}")
    print(f"RMS reprojection error: {result.rms_reprojection_error:.4f} pixels")
    print("Camera matrix:\n", result.camera_matrix)
    print("Distortion coefficients (k1, k2, p1, p2, k3):\n", result.distortion_coefficients)
    print(f"Saved {len(result.per_image_rms_pixels)} comparisons to {args.output_dir / 'comparisons'}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
