from __future__ import annotations

import socket
import struct
from datetime import datetime, timezone
from types import TracebackType
from typing import Callable

import numpy as np

from .models import RecordResult, TrignoConfig


class IncompleteReceiveError(ConnectionError):
    def __init__(self, message: str, partial: bytes) -> None:
        super().__init__(message)
        self.partial = partial


class TrignoClient:
    """Thin client preserving the repository's proven Trigno TCP protocol."""

    def __init__(
        self,
        config: TrignoConfig | None = None,
        socket_factory: Callable[..., socket.socket] = socket.create_connection,
    ) -> None:
        self.config = config or TrignoConfig()
        self._socket_factory = socket_factory
        self.command_socket: socket.socket | None = None
        self.data_socket: socket.socket | None = None
        self._streaming = False

    def connect(self) -> "TrignoClient":
        self.command_socket = self._socket_factory(
            (self.config.host, self.config.command_port), timeout=self.config.connect_timeout_s
        )
        try:
            self.data_socket = self._socket_factory(
                (self.config.host, self.config.emg_port), timeout=self.config.connect_timeout_s
            )
            self.command_socket.settimeout(1.0)
            try:
                self.command_socket.recv(1024)
            except socket.timeout:
                pass
            self.command_socket.settimeout(self.config.receive_timeout_s)
            self.data_socket.settimeout(self.config.receive_timeout_s)
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self) -> "TrignoClient":
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < nbytes:
            try:
                chunk = sock.recv(nbytes - received)
            except OSError as exc:
                raise IncompleteReceiveError(
                    f"Trigno exact receive failed after {received}/{nbytes} bytes: {exc}",
                    b"".join(chunks),
                ) from exc
            if not chunk:
                raise IncompleteReceiveError(
                    f"Trigno data socket disconnected after {received}/{nbytes} bytes",
                    b"".join(chunks),
                )
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    @staticmethod
    def decode_packet(packet: bytes, total_channels: int = 16, scale_to_mV: float = 1000.0) -> np.ndarray:
        frame_bytes = total_channels * 4
        if len(packet) == 0 or len(packet) % frame_bytes:
            raise ValueError("Trigno packet length must contain complete 16-channel float32 frames")
        sample_count = len(packet) // frame_bytes
        values = struct.unpack("<" + "f" * total_channels * sample_count, packet)
        return np.asarray(values, dtype=np.float32).reshape(sample_count, total_channels) * scale_to_mV

    def send_command(self, command: str) -> bytes:
        if self.command_socket is None:
            raise RuntimeError("Trigno command socket is not connected")
        self.command_socket.sendall((command + "\r\n\r\n").encode("ascii"))
        return self.command_socket.recv(128)

    def start(self) -> bytes:
        response = self.send_command("START")
        self._streaming = True
        return response

    def stop(self) -> bytes:
        if self.command_socket is None:
            return b""
        try:
            return self.send_command("STOP")
        finally:
            self._streaming = False

    def read_samples(self, sample_count: int) -> np.ndarray:
        if self.data_socket is None:
            raise RuntimeError("Trigno data socket is not connected")
        nbytes = sample_count * self.config.total_stream_channels * self.config.bytes_per_float
        packet = self.recv_exact(self.data_socket, nbytes)
        return self.decode_packet(packet, self.config.total_stream_channels, self.config.stream_scale_to_mV)

    def record(
        self,
        duration_s: float,
        channel_ids: tuple[int, ...] | list[int],
        progress_callback: Callable[[int], None] | None = None,
    ) -> RecordResult:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if not channel_ids or any(channel < 1 or channel > 16 for channel in channel_ids):
            raise ValueError("channel_ids must be explicit sensor IDs in 1..16")
        expected = int(round(duration_s * self.config.sample_rate_hz))
        chunks: list[np.ndarray] = []
        interrupted = False
        receive_error: str | None = None
        start_time = datetime.now(timezone.utc).isoformat()
        self.start()
        try:
            received = 0
            while received < expected:
                count = min(self.config.samples_per_read, expected - received)
                block = self.read_samples(count)
                chunks.append(block[:, [channel - 1 for channel in channel_ids]])
                received += block.shape[0]
                if progress_callback is not None:
                    progress_callback(received)
        except KeyboardInterrupt:
            interrupted = True
            receive_error = "KeyboardInterrupt"
        except IncompleteReceiveError as exc:
            frame_bytes = self.config.total_stream_channels * self.config.bytes_per_float
            complete_bytes = len(exc.partial) - (len(exc.partial) % frame_bytes)
            if complete_bytes:
                partial_block = self.decode_packet(
                    exc.partial[:complete_bytes],
                    self.config.total_stream_channels,
                    self.config.stream_scale_to_mV,
                )
                chunks.append(partial_block[:, [channel - 1 for channel in channel_ids]])
            receive_error = f"{type(exc).__name__}: {exc}"
        except (ConnectionError, OSError, socket.timeout) as exc:
            receive_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.stop()
            except (OSError, RuntimeError):
                self._streaming = False
        stop_time = datetime.now(timezone.utc).isoformat()
        emg = np.vstack(chunks) if chunks else np.empty((0, len(channel_ids)), dtype=np.float32)
        received = emg.shape[0]
        return RecordResult(
            emg_mV=emg,
            fs_hz=self.config.sample_rate_hz,
            stream_channel_ids=np.asarray(channel_ids, dtype=np.int16),
            expected_samples=expected,
            received_samples=received,
            dropped_samples=max(expected - received, 0),
            start_time=start_time,
            stop_time=stop_time,
            interrupted=interrupted,
            receive_error=receive_error,
        )

    def close(self) -> None:
        if self._streaming:
            try:
                self.stop()
            except (OSError, RuntimeError):
                pass
        for attr in ("data_socket", "command_socket"):
            sock = getattr(self, attr)
            if sock is not None:
                try:
                    sock.close()
                finally:
                    setattr(self, attr, None)
