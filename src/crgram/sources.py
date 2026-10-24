"""Where screenshots come from: a device over adb or an image file."""

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class SourceError(RuntimeError):
    pass


class Source(Protocol):
    @property
    def name(self) -> str: ...

    def capture(self) -> np.ndarray:
        """Return the current screen as a BGR image."""
        ...


class ImageSource:
    """A screenshot file; re-read on every capture so it can be replaced on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def name(self) -> str:
        return str(self.path)

    def capture(self) -> np.ndarray:
        try:
            data = np.fromfile(self.path, np.uint8)
        except OSError as e:
            raise SourceError(f"cannot read {self.path}: {e}") from e
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise SourceError(f"not an image: {self.path}")
        return image


class AdbSource:
    """Screen of an Android device. Foldables have several displays: by default
    the first one showing a non-black picture is used."""

    def __init__(self, serial: str | None = None, display: int | None = None) -> None:
        self.serial = serial
        self.display = display

    @property
    def name(self) -> str:
        return f"adb:{self.serial or 'default'}" + (f"#{self.display}" if self.display is not None else "")

    def _device(self):
        import adbutils

        try:
            if self.serial:
                return adbutils.adb.device(serial=self.serial)
            devices = adbutils.adb.device_list()
            if not devices:
                raise SourceError("adb: no devices connected")
            # A phone paired over Wi-Fi is often listed twice (mDNS name and ip:port).
            physical = {d.prop.get("ro.serialno") for d in devices}
            if len(physical) > 1:
                names = ", ".join(d.serial for d in devices)
                raise SourceError(f"adb: several devices connected ({names}), choose one with --serial")
            return devices[0]
        except adbutils.AdbError as e:
            raise SourceError(f"adb: {e}") from e

    def _shot(self, device, display: int | None) -> np.ndarray:
        import adbutils

        try:
            pil = device.screenshot(display_id=display, error_ok=False)
        except adbutils.AdbError as e:
            raise SourceError(f"screencap failed: {e}") from e
        return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    def capture(self) -> np.ndarray:
        device = self._device()
        if self.display is not None:
            return self._shot(device, self.display)
        last: SourceError | None = None
        for display in range(4):
            try:
                image = self._shot(device, display)
            except SourceError as e:
                last = e
                break
            if image.std() > 5:  # inactive display of a foldable is black
                self.display = display
                return image
        raise last or SourceError("no active display found")
