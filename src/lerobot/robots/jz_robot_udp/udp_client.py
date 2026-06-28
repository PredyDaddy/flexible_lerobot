#!/usr/bin/env python

from __future__ import annotations

import logging
import socket
import threading

from .protocol import ProtocolError, decode_state_packet
from .state_cache import StateCache

logger = logging.getLogger(__name__)


class UDPStateReceiver:
    """Background UDP receiver for readonly JZRobot state packets."""

    def __init__(self, bind_ip: str, port: int, cache: StateCache, buffer_size: int = 65535):
        self.bind_ip = bind_ip
        self.port = port
        self.cache = cache
        self.buffer_size = buffer_size
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind_ip, self.port))
        sock.settimeout(0.2)
        self._socket = sock
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="jz_robot_udp_state_receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                return
            try:
                data, sender = sock.recvfrom(self.buffer_size)
            except TimeoutError:
                continue
            except OSError:
                if not self._stop_event.is_set():
                    logger.exception("UDP state receiver socket error")
                return

            try:
                packet = decode_state_packet(data)
            except ProtocolError:
                logger.exception("Ignoring invalid UDP state packet from %s", sender)
                continue
            self.cache.update(packet, sender)


class UDPCommandSender:
    """UDP sender for Phase 2 command dry-run packets.

    This helper only sends bytes to a configured target. It does not receive state,
    publish ROS messages, or encode any robot execution semantics.
    """

    def __init__(self, target_ip: str, target_port: int, timeout_s: float = 0.2):
        self.target_ip = target_ip
        self.target_port = target_port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None

    def send(self, data: bytes) -> int:
        sock = self._socket
        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.timeout_s)
            self._socket = sock
        return sock.sendto(data, (self.target_ip, self.target_port))

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
