"""Capture one banana-leaf photo from the Raspberry Pi camera.

picamera2 is used when it is installed. Otherwise the system `rpicam-still`
or `libcamera-still` command is used, which does not have to live inside the
Python virtual environment.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


def capture_banana_leaf(dest: Path, preview: bool = True, warmup_seconds: float = 2.0) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    if _picamera2_importable():
        try:
            _capture_picamera2(dest, preview=preview, warmup_seconds=warmup_seconds)
            print(f"Captured: {dest.resolve()}")
            return dest
        except Exception as exc:
            errors.append(f"picamera2: {exc}")
            print(f"WARNING: picamera2 could not capture ({exc}). Trying rpicam-still.")

    try:
        _capture_rpicam(dest, preview=preview, warmup_seconds=warmup_seconds)
    except Exception as exc:
        errors.append(f"rpicam-still: {exc}")
        detail = "\n".join(errors)
        raise SystemExit(
            "Could not open the Raspberry Pi camera.\n"
            f"{detail}\n"
            "Install the camera tools with: sudo apt install -y rpicam-apps\n"
            "Then check the cable and detection: rpicam-hello --list-cameras"
        ) from exc

    print(f"Captured: {dest.resolve()}")
    return dest


def _picamera2_importable() -> bool:
    try:
        import picamera2  # noqa: F401
    except Exception:
        return False
    return True


def _wait_to_frame(warmup_seconds: float, preview: bool) -> None:
    time.sleep(max(0.5, warmup_seconds))
    if sys.stdin.isatty():
        where = "in the preview" if preview else "in front of the camera"
        input(f"Frame the banana leaf {where}, then press Enter to capture... ")


def _capture_picamera2(dest: Path, preview: bool, warmup_seconds: float) -> None:
    from picamera2 import Picamera2

    camera = Picamera2()
    camera.configure(camera.create_still_configuration())
    try:
        if preview:
            try:
                camera.start(show_preview=True)
            except Exception as exc:
                print(f"WARNING: camera preview is unavailable ({exc}). Capturing without a preview.")
                camera.start()
                preview = False
        else:
            camera.start()
        _wait_to_frame(warmup_seconds, preview)
        camera.capture_file(str(dest))
    finally:
        try:
            camera.stop()
        finally:
            camera.close()
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"picamera2 did not write an image to {dest}")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _capture_rpicam(dest: Path, preview: bool, warmup_seconds: float) -> None:
    binary = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if binary is None:
        raise FileNotFoundError("rpicam-still and libcamera-still were not found on PATH")

    hello = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    preview_proc = None
    if preview and hello is not None and sys.stdin.isatty():
        preview_proc = subprocess.Popen([hello, "-t", "0"])
        try:
            input("Frame the banana leaf in the preview, then press Enter to capture... ")
        finally:
            _stop_process(preview_proc)
        # The preview has released the camera. A short timeout lets exposure settle.
        timeout_ms = max(500, int(warmup_seconds * 1000))
        show_still_preview = False
    elif sys.stdin.isatty():
        where = "in the preview" if preview else "in front of the camera"
        print(f"Frame the banana leaf {where}. The photo is taken when the timer ends.")
        timeout_ms = max(1000, int(max(warmup_seconds, 5.0) * 1000))
        show_still_preview = preview
    else:
        timeout_ms = max(1000, int(warmup_seconds * 1000))
        print(f"No keyboard attached. Capturing after {timeout_ms / 1000:.1f}s.")
        show_still_preview = preview

    command = [binary, "-o", str(dest), "-t", str(timeout_ms)]
    if not show_still_preview:
        command.append("--nopreview")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{Path(binary).name} exited with status {completed.returncode}")
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"{Path(binary).name} did not write an image to {dest}")
