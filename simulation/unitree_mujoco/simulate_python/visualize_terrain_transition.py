"""Interactive and offscreen visualization for frozen Terrain Transition v1.

This runner never creates a simplified scene.  It calls ``run_one`` from the
authoritative transition module and attaches a read-only observer for rendering,
wall-clock pacing, terminal status, and optional MP4 capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import threading
import time

import mujoco
import numpy as np

from run_terrain_transition import CASES, Run, TransitionObserverFrame, run_one


SIM_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RECORD_DIRECTORY = SIM_DIR / "outputs" / "terrain_transition_visualization"
TERRAIN_RGBA = {
    "concrete": np.asarray((0.38, 0.40, 0.42, 1.0), dtype=np.float32),
    "marble": np.asarray((0.72, 0.72, 0.76, 1.0), dtype=np.float32),
    "ice": np.asarray((0.62, 0.84, 0.96, 1.0), dtype=np.float32),
    "sand": np.asarray((0.76, 0.57, 0.30, 1.0), dtype=np.float32),
}


def apply_visual_terrain_appearance(model: mujoco.MjModel, terrain: str) -> None:
    """Set only RGBA values on both transition grounds; no physics fields change."""
    if terrain not in TERRAIN_RGBA:
        raise ValueError(f"unknown terrain appearance: {terrain}")
    for ground in ("ground_a", "ground_b"):
        model.geom_rgba[model.geom(ground).id] = TERRAIN_RGBA[terrain]


class TransitionVisualObserver:
    """Visual-only observer used by the authoritative transition stepping loop."""

    def __init__(
        self,
        *,
        show_viewer: bool,
        speed: float,
        hold_seconds: float,
        record_path: Path | None,
        record_fps: int,
        width: int,
        height: int,
    ) -> None:
        if speed <= 0.0 or hold_seconds < 0.0:
            raise ValueError("speed must be positive and hold_seconds must not be negative")
        if record_fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("record fps and dimensions must be positive")
        self.show_viewer = show_viewer
        self.speed = speed
        self.hold_seconds = hold_seconds
        self.record_path = record_path
        self.record_fps = record_fps
        self.width = width
        self.height = height
        self.viewer = None
        self.renderer: mujoco.Renderer | None = None
        self.frames: list[np.ndarray] = []
        self.next_frame_s = 0.0
        self.next_sync_s = 0.0
        self.wall_start: float | None = None
        self.simulation_start = 0.0
        self.viewer_threads: list[threading.Thread] = []
        self.completed = False
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.show_viewer or self.record_path is not None

    def _configure_camera(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Use a fixed free camera that includes the foot and both ground halves."""
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
                self.viewer.cam.distance = 4.2
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.elevation = -24.0

    @staticmethod
    def _record_camera() -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.35)
        camera.distance = 4.2
        camera.azimuth = 135.0
        camera.elevation = -24.0
        return camera

    def _initialize(self, model: mujoco.MjModel, data: mujoco.MjData, frame: TransitionObserverFrame) -> None:
        apply_visual_terrain_appearance(model, frame.terrain)
        self.simulation_start = float(data.time)
        self.wall_start = time.perf_counter()
        if self.show_viewer:
            from mujoco import viewer as mujoco_viewer

            before = set(threading.enumerate())
            self.viewer = mujoco_viewer.launch_passive(model, data)
            self.viewer_threads = [thread for thread in threading.enumerate() if thread not in before]
            self._configure_camera(model, data)
        if self.record_path is not None:
            self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        print(
            f"CASE {frame.case_id} | {data.time * 1000.0:.0f} ms | "
            f"terrain={frame.terrain.upper()} | T0={self._transition_text(frame.case_id)}"
        )

    @staticmethod
    def _transition_text(case_id: str) -> str:
        case = CASES[case_id]
        return f"{case['before'].upper()}→{case['after'].upper()} @ 650 ms"

    def _capture_if_due(self, data: mujoco.MjData) -> None:
        if self.renderer is None or data.time + 1e-12 < self.next_frame_s:
            return
        self.renderer.update_scene(data, camera=self._record_camera())
        self.frames.append(self.renderer.render().copy())
        self.next_frame_s += 1.0 / self.record_fps

    def _sync_and_pace(self, data: mujoco.MjData) -> None:
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed before the transition completed")
        if data.time + 1e-12 >= self.next_sync_s:
            self.viewer.sync()
            self.next_sync_s += 1.0 / 60.0
        assert self.wall_start is not None
        target_elapsed = (float(data.time) - self.simulation_start) / self.speed
        remaining = target_elapsed - (time.perf_counter() - self.wall_start)
        if remaining > 0.0:
            time.sleep(remaining)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, frame: TransitionObserverFrame) -> None:
        if frame.phase == "initialize":
            self._initialize(model, data, frame)
            return
        if frame.phase == "transition":
            # The callback occurs in the exact core tick immediately after the
            # contact profile changes, before the next mj_step.
            apply_visual_terrain_appearance(model, frame.terrain)
            print(f"CASE {frame.case_id} | {data.time * 1000.0:.0f} ms | T0 → {frame.terrain.upper()}")
            return
        if frame.phase == "step":
            self._capture_if_due(data)
            self._sync_and_pace(data)
        elif frame.phase == "finish":
            self.completed = True

    def _write_mp4(self) -> None:
        assert self.record_path is not None
        if not self.frames:
            raise RuntimeError("recording produced no frames")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("--record requires ffmpeg on PATH")
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        if self.record_path.exists():
            raise FileExistsError(f"refusing to overwrite recording: {self.record_path}")
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}", "-r", str(self.record_fps),
            "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(self.record_path),
        ]
        process = subprocess.run(command, input=b"".join(frame.tobytes() for frame in self.frames), check=False)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg recording failed with exit code {process.returncode}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.record_path is not None and self.completed:
                self._write_mp4()
        finally:
            if self.renderer is not None:
                self.renderer.close()
            if self.viewer is not None:
                if self.completed and self.hold_seconds > 0.0:
                    print(f"Holding final frame for {self.hold_seconds:g} s; close the viewer to exit early.")
                    deadline = time.perf_counter() + self.hold_seconds
                    while self.viewer.is_running() and time.perf_counter() < deadline:
                        self.viewer.sync()
                        time.sleep(0.02)
                self.viewer.close()
                for thread in self.viewer_threads:
                    thread.join()


def run_visualization(
    case_id: str,
    *,
    run_index: int = 0,
    family: str = "multisine",
    surface_index: int | None = None,
    viewer: bool = True,
    speed: float = 1.0,
    hold_seconds: float = 5.0,
    record_path: Path | None = None,
    record_fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Run:
    """Run one case using the same core physics as headless materialization."""
    if case_id not in CASES:
        raise ValueError(f"case must be one of {', '.join(CASES)}")
    if record_path is not None and record_path.exists():
        raise FileExistsError(f"refusing to overwrite recording: {record_path}")
    observer = None
    visual = None
    if viewer or record_path is not None:
        visual = TransitionVisualObserver(
            show_viewer=viewer, speed=speed, record_path=record_path,
            hold_seconds=hold_seconds,
            record_fps=record_fps, width=width, height=height,
        )
        observer = visual
    try:
        return run_one(case_id, run_index, family, surface_index, observer=observer)
    finally:
        if visual is not None:
            visual.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True, help="transition case A/B/C/D")
    parser.add_argument("--run-index", type=int, default=0, help="deterministic representative run index")
    parser.add_argument("--surface-family", default="multisine", help="existing expanded-dataset surface family")
    parser.add_argument("--surface-index", type=int, help="surface realization; defaults to --run-index")
    parser.set_defaults(viewer=True)
    parser.add_argument("--viewer", dest="viewer", action="store_true", help="open the interactive MuJoCo viewer (default)")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false", help="run the same visual runner without a GUI")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer wall-clock playback multiplier; physics remains 2 kHz")
    parser.add_argument("--hold-seconds", type=float, default=5.0, help="keep the final viewer frame visible after physics completes; 0 disables")
    parser.add_argument("--record", action="store_true", help="write an offscreen MP4 using ffmpeg")
    parser.add_argument("--output", type=Path, help="MP4 path; default is outputs/terrain_transition_visualization/case_<id>.mp4")
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)
    if args.run_index < 0 or args.surface_index is not None and args.surface_index < 0:
        parser.error("run and surface indices must be non-negative")
    if args.speed <= 0.0 or args.hold_seconds < 0.0:
        parser.error("--speed must be positive and --hold-seconds must not be negative")
    if args.record_fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("record fps and dimensions must be positive")
    if args.output is not None and not args.record:
        parser.error("--output requires --record")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record_path = None
    if args.record:
        record_path = (args.output or DEFAULT_RECORD_DIRECTORY / f"case_{args.case.lower()}.mp4").resolve()
    run = run_visualization(
        args.case, run_index=args.run_index, family=args.surface_family,
        surface_index=args.surface_index, viewer=args.viewer, speed=args.speed,
        hold_seconds=args.hold_seconds,
        record_path=record_path, record_fps=args.record_fps, width=args.width, height=args.height,
    )
    print(
        f"CASE {args.case} COMPLETE | samples={len(run.time_s)} | "
        f"T0={run.metadata['transition_t0_ms']:.0f} ms | "
        f"final_time={run.time_s[-1] * 1000.0:.0f} ms"
    )
    if record_path is not None:
        print(f"RECORDING={record_path}")


if __name__ == "__main__":
    main()
