import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lane_fitting import LaneDetection, LaneSettings, detect_lane_boundaries
from lane_metrics import MetricSettings, measure_lane_geometry
from lane_overlay import create_lane_masks, project_lane, render_readout
from perspective_transform import PerspectiveSettings, create_perspective_transform, warp_binary
from thresholding import ThresholdSettings, or_threshold, save_image
from undistort_images import load_calibration, undistort_image


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrackingSettings:
    smoothing_alpha: float = 0.25
    maximum_shift_fraction: float = 0.1
    maximum_hold_frames: int = 5
    reacquire_frames: int = 3

    def __post_init__(self):
        if not np.isfinite([self.smoothing_alpha, self.maximum_shift_fraction]).all():
            raise ValueError("Tracking settings must be finite.")
        if not 0 < self.smoothing_alpha <= 1 or not 0 < self.maximum_shift_fraction < 1:
            raise ValueError("Use a smoothing alpha in (0, 1] and a shift fraction in (0, 1).")
        if type(self.maximum_hold_frames) is not int or self.maximum_hold_frames < 0:
            raise ValueError("Maximum hold frames must be a nonnegative integer.")
        if type(self.reacquire_frames) is not int or self.reacquire_frames < 1:
            raise ValueError("Reacquisition requires a positive integer frame count.")


@dataclass
class TrackedLane:
    coefficients: np.ndarray | None
    status: str
    held_frames: int
    issues: list[str]


class LaneTracker:
    def __init__(self, image_size: tuple[int, int], settings: TrackingSettings):
        self.image_size = image_size
        self.settings = settings
        self.coefficients = None
        self.pending = None
        self.pending_count = 0
        self.missed = 0
        self.ever_detected = False
        self.rows = np.linspace(0, image_size[1] - 1, 25)

    def shift(self, first: np.ndarray, second: np.ndarray) -> float:
        return max(float(np.max(np.abs(np.polyval(a - b, self.rows)))) for a, b in zip(first, second))

    def update(self, detection: LaneDetection) -> TrackedLane:
        issues = list(detection.issues)
        candidate = None
        if not issues and detection.left is not None and detection.right is not None:
            candidate = np.array([detection.left.coefficients, detection.right.coefficients], dtype=np.float64)
            if candidate.shape != (2, 3) or not np.isfinite(candidate).all():
                candidate = None
                issues.append("Invalid lane coefficients.")
        if candidate is not None:
            limit = self.image_size[0] * self.settings.maximum_shift_fraction
            if self.coefficients is None:
                status = "reacquired" if self.ever_detected else "detected"
                self.coefficients = candidate
            elif self.shift(candidate, self.coefficients) <= limit:
                alpha = self.settings.smoothing_alpha
                self.coefficients = alpha * candidate + (1 - alpha) * self.coefficients
                status = "detected"
            else:
                if self.pending is not None and self.shift(candidate, self.pending) <= limit / 2:
                    self.pending_count += 1
                else:
                    self.pending_count = 1
                self.pending = candidate
                if self.pending_count >= self.settings.reacquire_frames:
                    self.coefficients = candidate
                    status = "reacquired"
                else:
                    status = None
                    issues.append("Abrupt lane shift awaiting consistent detections.")
            if status is not None:
                self.missed = 0
                self.pending = None
                self.pending_count = 0
                self.ever_detected = True
                return TrackedLane(self.coefficients.copy(), status, 0, issues)
        else:
            self.pending = None
            self.pending_count = 0
            if not issues:
                issues.append("Both lane boundaries are required.")
        self.missed += 1
        if self.coefficients is not None and self.missed <= self.settings.maximum_hold_frames:
            return TrackedLane(self.coefficients.copy(), "held", self.missed, issues)
        self.coefficients = None
        return TrackedLane(None, "unavailable", self.missed, issues)


class FramePipeline:
    def __init__(
        self, image_size: tuple[int, int], calibration: tuple,
        metric_settings: MetricSettings, tracking_settings: TrackingSettings,
    ):
        self.image_size = image_size
        self.calibration = calibration
        self.metric_settings = metric_settings
        self.threshold_settings = ThresholdSettings()
        self.lane_settings = LaneSettings()
        self.transform = create_perspective_transform(image_size, PerspectiveSettings())
        self.tracker = LaneTracker(image_size, tracking_settings)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be unsigned 8-bit BGR images.")
        if frame.shape[1::-1] != self.image_size:
            raise ValueError("Video frame resolution changed during processing.")
        corrected = undistort_image(frame, *self.calibration)
        binary = or_threshold(corrected, self.threshold_settings)
        warped = warp_binary(binary, self.transform.matrix)
        detection = detect_lane_boundaries(warped, self.lane_settings)
        tracked = self.tracker.update(detection)
        overlay = corrected
        metrics = None
        if tracked.coefficients is not None:
            left, right = tracked.coefficients
            masks = create_lane_masks(left, right, self.image_size)
            overlay = project_lane(corrected, masks, self.transform.inverse_matrix)
            metrics = measure_lane_geometry(
                left, right, self.image_size, self.transform.destination_points,
                self.transform.inverse_matrix, self.metric_settings,
            )
        output = render_readout(overlay, metrics, self.metric_settings)
        label = f"Tracking: {tracked.status}"
        if tracked.status == "held":
            label += f" ({tracked.held_frames}/{self.tracker.settings.maximum_hold_frames} frames; last accepted fit)"
        scale = self.image_size[0] / 1280
        cv2.putText(
            output, label, (round(28 * scale), round(198 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
            (0, 200, 255) if tracked.status in ("held", "unavailable") else (255, 255, 255),
            1, cv2.LINE_AA,
        )
        return output, {
            "status": tracked.status, "held_frames": tracked.held_frames,
            "issues": tracked.issues,
            "left_coefficients": None if tracked.coefficients is None else tracked.coefficients[0].tolist(),
            "right_coefficients": None if tracked.coefficients is None else tracked.coefficients[1].tolist(),
            "center_radius_m": None if metrics is None else metrics["radius_m"]["center"],
            "straight": None if metrics is None else metrics["straight"]["center"],
            "vehicle_offset_m": None if metrics is None else metrics["vehicle_offset_m"],
        }


def process_video(
    input_path: Path, output_dir: Path, calibration: tuple,
    metric_settings: MetricSettings, tracking_settings: TrackingSettings,
) -> dict:
    input_path = input_path.resolve()
    output_path = (output_dir / f"{input_path.stem}_output.mp4").resolve()
    if output_path == input_path:
        raise ValueError("Input and output video paths must differ.")
    capture = cv2.VideoCapture(str(input_path))
    writer = None
    start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise ValueError(f"Invalid frame rate or dimensions: {input_path}")
        if width % 2 or height % 2:
            raise ValueError("MP4 output requires even frame dimensions to avoid implicit cropping.")
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot open MP4 writer: {output_path}")
        pipeline = FramePipeline((width, height), calibration, metric_settings, tracking_settings)
        records = []
        preview_indices = set(np.linspace(0, max(0, expected_frames - 1), 6, dtype=int).tolist())
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, record = pipeline.process(frame)
            index = len(records)
            writer.write(output)
            record.update({"frame": index, "time_seconds": index / fps})
            records.append(record)
            if index in preview_indices:
                save_image(output_dir / "previews" / f"{input_path.stem}_{index:05d}.jpg", output)
            if len(records) % 250 == 0:
                print(f"{input_path.name}: {len(records)}/{expected_frames} frames", flush=True)
        if not records:
            raise ValueError(f"No readable frames: {input_path}")
        if expected_frames > 0 and len(records) != expected_frames:
            raise ValueError(f"Decoded {len(records)} of {expected_frames} expected frames: {input_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    verification = cv2.VideoCapture(str(output_path))
    try:
        actual_size = (int(verification.get(cv2.CAP_PROP_FRAME_WIDTH)), int(verification.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        encoded_frames = int(verification.get(cv2.CAP_PROP_FRAME_COUNT))
        encoded_fps = float(verification.get(cv2.CAP_PROP_FPS))
        if not verification.isOpened() or actual_size != (width, height) or encoded_frames != len(records):
            raise OSError(f"Output video metadata verification failed: {output_path}")
        if abs(encoded_fps - fps) > 0.001:
            raise OSError(f"Output frame rate verification failed: {output_path}")
    finally:
        verification.release()
    report = {
        "input": str(input_path), "output": output_path.name,
        "image_size": [width, height], "fps": fps, "frame_count": len(records),
        "duration_seconds": len(records) / fps, "processing_seconds": time.perf_counter() - start,
        "codec": "mp4v", "audio": "Output contains video only.",
        "metric_settings": asdict(metric_settings), "tracking_settings": asdict(tracking_settings),
        "threshold_settings": asdict(pipeline.threshold_settings),
        "lane_settings": asdict(pipeline.lane_settings),
        "perspective_settings": asdict(PerspectiveSettings()),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "frames": records,
    }
    (output_dir / f"{input_path.stem}_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Finished {input_path.name}: {report['status_counts']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete lane-finding pipeline on road videos.")
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--videos-dir", type=Path, default=PROJECT_DIR / "test_videos")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/video")
    parser.add_argument("--calibration", type=Path, default=PROJECT_DIR / "outputs/calibration/camera_calibration.npz")
    parser.add_argument("--metrics-report", type=Path, default=PROJECT_DIR / "outputs/lane_metrics/lane_metrics_report.json")
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--max-hold-frames", type=int, default=5)
    args = parser.parse_args()
    paths = args.input if args.input else sorted(args.videos_dir.glob("*.mp4"))
    if not paths:
        raise ValueError(f"No MP4 videos found in {args.videos_dir}.")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Input videos must have unique filenames to prevent output collisions.")
    if any((args.output_dir / f"{path.stem}_output.mp4").resolve() == path.resolve() for path in paths):
        raise ValueError("Input and output video paths must differ.")
    calibration = load_calibration(args.calibration)
    metric_settings = MetricSettings(**json.loads(args.metrics_report.read_text(encoding="utf-8"))["settings"])
    tracking_settings = TrackingSettings(smoothing_alpha=args.smoothing_alpha, maximum_hold_frames=args.max_hold_frames)
    reports = [process_video(path, args.output_dir, calibration, metric_settings, tracking_settings) for path in paths]
    gallery = [
        "# Lane-finding video results", "",
        "Outputs retain the input frame rate, resolution, and frame count. Audio is omitted.",
        "Green area, cyan left boundary, red right boundary. Metrics use the previous step's assumptions.",
        "Held fits are explicitly labeled; unavailable frames display no lane overlay.", "",
        "| Video | Frames | Detected | Reacquired | Held | Unavailable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        counts = report["status_counts"]
        values = " | ".join(str(counts.get(status, 0)) for status in ("detected", "reacquired", "held", "unavailable"))
        gallery.append(f"| [{Path(report['input']).name}]({report['output']}) | {report['frame_count']} | {values} |")
    for report in reports:
        stem = Path(report["input"]).stem
        gallery.extend(("", f"## {stem}", "", f"[Annotated video]({report['output']}) · [Frame report]({stem}_report.json)", ""))
        for preview in sorted((args.output_dir / "previews").glob(f"{stem}_*.jpg")):
            gallery.extend((f"![{preview.stem}](previews/{preview.name})", ""))
    (args.output_dir / "gallery.md").write_text("\n".join(gallery) + "\n", encoding="utf-8")
    print(f"Gallery: {args.output_dir / 'gallery.md'}")


if __name__ == "__main__":
    main()
