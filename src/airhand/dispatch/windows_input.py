"""Moving the Windows cursor and reading a hotkey, through the Win32 API (ctypes).

Why absolute moves: a relative SendInput move (MOUSEEVENTF_MOVE alone) is passed through the
Windows pointer-speed setting and "Enhance pointer precision" acceleration, which would stack
a second, unknown gain curve on top of ours. Instead we read the current cursor position,
add our move, keep it on the screen, and send an ABSOLUTE move. Then our transfer function
is the only one applied.

Absolute SendInput coordinates are normalised to 0..65535 across the virtual desktop (all
monitors together, which may start at negative coordinates).

The process is made per-monitor DPI aware first; otherwise Windows scales the coordinates we
read and write on high-DPI screens.

Limitation: Windows does not let a normal process send input to windows of an elevated
(administrator) process (User Interface Privilege Isolation).

`RecordingMouse` has the same interface and just records moves, for tests and dry runs.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from typing import Protocol

# Virtual-key codes for the function keys: VK_F1 = 0x70 ... VK_F12 = 0x7B.
VK_F1 = 0x70


def virtual_key_code(name: str) -> int:
    """Win32 virtual-key code for a hotkey name such as "F8"."""
    upper = name.strip().upper()
    if upper.startswith("F") and upper[1:].isdigit() and 1 <= int(upper[1:]) <= 12:
        return VK_F1 + int(upper[1:]) - 1
    raise ValueError(f"Unsupported hotkey {name!r}; use F1..F12")


@dataclass(frozen=True)
class ScreenRect:
    left: int
    top: int
    width: int
    height: int

    def clamp(self, x: int, y: int) -> tuple[int, int]:
        x = min(max(x, self.left), self.left + self.width - 1)
        y = min(max(y, self.top), self.top + self.height - 1)
        return x, y

    def to_absolute(self, x: int, y: int) -> tuple[int, int]:
        """Pixel position -> SendInput absolute coordinates (0..65535)."""
        ax = round((x - self.left) * 65535 / max(self.width - 1, 1))
        ay = round((y - self.top) * 65535 / max(self.height - 1, 1))
        return ax, ay


class MouseOutput(Protocol):
    def move_by(self, dx: int, dy: int) -> None: ...


@dataclass
class RecordingMouse:
    """Stand-in mouse that remembers where the cursor would be."""

    screen: ScreenRect
    x: int
    y: int
    path: list[tuple[int, int]] = field(default_factory=list)

    def move_by(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self.x, self.y = self.screen.clamp(self.x + dx, self.y + dy)
        self.path.append((self.x, self.y))


# --- Win32 -----------------------------------------------------------------------------

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    # The keyboard member is listed only so the union has its real (largest) size.
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


def _user32() -> ctypes.WinDLL:
    if sys.platform != "win32":
        raise OSError("Windows input injection is only available on Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


class WindowsMouse:
    """Moves the real cursor with absolute SendInput moves."""

    def __init__(self) -> None:
        self.user32 = _user32()
        self.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
        self.user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_INPUT), ctypes.c_int]
        self.screen = ScreenRect(
            left=self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            top=self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            width=self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            height=self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )

    def position(self) -> tuple[int, int]:
        point = _POINT()
        if not self.user32.GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return point.x, point.y

    def move_by(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        x, y = self.position()
        x, y = self.screen.clamp(x + dx, y + dy)
        ax, ay = self.screen.to_absolute(x, y)
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        event = _INPUT(type=INPUT_MOUSE, u=_INPUTUNION(mi=_MOUSEINPUT(ax, ay, 0, flags, 0, 0)))
        if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())


class HotkeyPoller:
    """Reports each new press of one key, by polling its state (no keyboard hook needed).

    GetAsyncKeyState's high bit is set while the key is down; a press is a transition from
    up to down between two polls.
    """

    def __init__(self, key_name: str) -> None:
        self.user32 = _user32()
        self.vk = virtual_key_code(key_name)
        self._was_down = self._is_down()

    def _is_down(self) -> bool:
        return bool(self.user32.GetAsyncKeyState(self.vk) & 0x8000)

    def pressed(self) -> bool:
        down = self._is_down()
        new_press = down and not self._was_down
        self._was_down = down
        return new_press
