import sys
import types
import subprocess
from pathlib import Path

import numpy as np

from loco_mujoco.core.visuals.video_recorder import VideoRecorder


def test_stop_is_idempotent(tmp_path, monkeypatch):
    release_calls = {"n": 0}

    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            release_calls["n"] += 1
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0
    cv2.VideoWriter = lambda *_args, **_kwargs: DummyWriter()
    cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    recorder(np.zeros((8, 8, 3), dtype=np.uint8))

    path1 = recorder.stop()
    path2 = recorder.stop()

    assert path1 is not None
    assert path2 == path1
    assert release_calls["n"] == 1


def test_stop_without_frames_returns_none(tmp_path):
    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=30,
        compress=False,
    )
    assert recorder.stop() is None


def test_compression_timeout_keeps_original_video(tmp_path, monkeypatch):
    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0

    def make_writer(path, *_args, **_kwargs):
        with open(path, "wb") as stream:
            stream.write(b"original-mp4")
        return DummyWriter()

    cv2.VideoWriter = make_writer
    cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    calls = []

    def timeout_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout_run)
    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=100,
        compress=True,
    )
    recorder(np.zeros((8, 8, 3), dtype=np.uint8))

    output = recorder.stop()

    assert Path(output).read_bytes() == b"original-mp4"
    assert not (tmp_path / "test" / "tmp_recording.mp4").exists()
    command, kwargs = calls[0]
    assert command[command.index("-threads") + 1] == "2"
    assert kwargs["timeout"] == 120


def test_successful_compression_atomically_replaces_original(tmp_path, monkeypatch):
    class DummyWriter:
        def write(self, _frame):
            return None

        def release(self):
            return None

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_RGB2BGR = 0
    cv2.cvtColor = lambda frame, _code: frame
    cv2.VideoWriter_fourcc = lambda *_args: 0

    def make_writer(path, *_args, **_kwargs):
        Path(path).write_bytes(b"original-mp4")
        return DummyWriter()

    cv2.VideoWriter = make_writer
    cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    def successful_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"h264-mp4")

    monkeypatch.setattr(subprocess, "run", successful_run)
    recorder = VideoRecorder(
        path=str(tmp_path),
        tag="test",
        video_name="recording",
        fps=100,
        compress=True,
    )
    recorder(np.zeros((8, 8, 3), dtype=np.uint8))

    output = recorder.stop()

    assert Path(output).read_bytes() == b"h264-mp4"
    assert not (tmp_path / "test" / "tmp_recording.mp4").exists()
