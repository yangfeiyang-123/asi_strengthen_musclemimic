"""Deprecated compatibility surface for the original single-file collector.

New experiments must use ``python -m emg.cli``.  Importing this module has no
filesystem, timestamp, plotting, or hardware side effects.
"""

from __future__ import annotations

import socket
import sys
import warnings

import numpy as np

from emg.models import ProcessingConfig, TrignoConfig
from emg.processing import emg_envelope as _emg_envelope
from emg.trigno import TrignoClient


_config = TrignoConfig()
HOST = _config.host
CMD_PORT = _config.command_port
EMG_PORT = _config.emg_port
FS = int(_config.sample_rate_hz)
TOTAL_CHANNELS = _config.total_stream_channels
BYTES_PER_FLOAT = _config.bytes_per_float
samples_per_read = _config.samples_per_read
stream_scale_to_mV = _config.stream_scale_to_mV

# Read-only compatibility defaults. New code uses ChannelProfile objects.
sensor_channels = [1]
sensor_indices = [0]
channel_names = {1: "Gastrocnemius"}


def send_cmd(sock: socket.socket, cmd: str) -> bytes:
    sock.sendall((cmd + "\r\n\r\n").encode("ascii"))
    return sock.recv(128)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    return TrignoClient.recv_exact(sock, nbytes)


def read_emg_block(data_sock: socket.socket, sample_count: int) -> np.ndarray:
    packet = recv_exact(data_sock, sample_count * TOTAL_CHANNELS * BYTES_PER_FLOAT)
    # Preserve the original [channels, samples] compatibility shape.
    return TrignoClient.decode_packet(packet, TOTAL_CHANNELS, stream_scale_to_mV).T


def emg_envelope(emg_arr: np.ndarray) -> np.ndarray:
    _, envelope = _emg_envelope(emg_arr, ProcessingConfig(normalization="none"), 10.0)
    return envelope


def main() -> int:
    warnings.warn(
        "delysis_measure.py is deprecated; use `python -m emg.cli mvc --help`.",
        DeprecationWarning,
        stacklevel=2,
    )
    from emg.cli import main as cli_main

    return cli_main(["mvc", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
