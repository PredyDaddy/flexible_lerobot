from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SoundStatus:
    enabled: bool
    initialized: bool
    init_error: str | None
    sounds_loaded: list[str]


class SoundPlayer:
    """Best-effort server-side audio playback using pygame.

    Why server-side:
    - Browser autoplay policies often block audio.
    - We want the robot PC to speak regardless of client device/browser.
    """

    def __init__(self, *, sounds_dir: Path, files: dict[str, str], enabled: bool = True, volume: float = 1.0):
        self._enabled_cfg = bool(enabled)
        self._volume = float(volume)
        self._sounds_dir = Path(sounds_dir).expanduser().resolve()
        self._files_cfg = dict(files)

        self._pygame = None
        self._sounds: dict[str, object] = {}
        self._init_error: str | None = None
        self._initialized = False

        self._q: queue.Queue[tuple[str, bool, threading.Event]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="SoundPlayer", daemon=True)

        self._init_pygame()
        self._thread.start()

    def _init_pygame(self) -> None:
        if not self._enabled_cfg:
            self._init_error = "disabled_by_config"
            self._initialized = False
            return

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame  # type: ignore

            # mixer.init can fail on headless systems / missing audio devices. We keep running without sound.
            pygame.mixer.init()
            self._pygame = pygame
            self._sounds = {}

            for key, fname in self._files_cfg.items():
                path = (self._sounds_dir / fname).resolve()
                if not path.is_file():
                    continue
                snd = pygame.mixer.Sound(str(path))
                try:
                    snd.set_volume(self._volume)
                except Exception:
                    # Older pygame versions may not support per-sound volume; ignore.
                    pass
                self._sounds[str(key)] = snd

            self._initialized = True
            self._init_error = None
        except Exception as e:
            self._pygame = None
            self._sounds = {}
            self._initialized = False
            self._init_error = f"{type(e).__name__}: {e}"

    def status(self) -> SoundStatus:
        enabled = self._enabled_cfg and self._initialized and bool(self._sounds)
        return SoundStatus(
            enabled=enabled,
            initialized=self._initialized,
            init_error=self._init_error,
            sounds_loaded=sorted(self._sounds.keys()),
        )

    def play(self, name: str, *, block: bool = False) -> bool:
        st = self.status()
        if not st.enabled:
            return False

        ev = threading.Event()
        self._q.put((str(name), bool(block), ev))
        if block:
            ev.wait(timeout=10.0)
        return True

    def _worker(self) -> None:
        while True:
            name, block, ev = self._q.get()
            try:
                snd = self._sounds.get(name)
                if snd is None:
                    continue

                # Avoid overlapping prompts. This keeps the UX clear.
                try:
                    assert self._pygame is not None
                    self._pygame.mixer.stop()
                except Exception:
                    pass

                ch = None
                try:
                    ch = snd.play()
                except Exception:
                    ch = None

                if block and ch is not None:
                    t0 = time.time()
                    while ch.get_busy() and time.time() - t0 < 10.0:
                        time.sleep(0.02)
            finally:
                ev.set()

