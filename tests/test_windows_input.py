"""Tests for cursor output helpers (no real cursor movement)."""

from __future__ import annotations

import ctypes
import sys

import pytest

from airhand.dispatch.windows_input import (
    _INPUT,
    HotkeyPoller,
    RecordingMouse,
    ScreenRect,
    WindowsMouse,
    virtual_key_code,
)

DUAL = ScreenRect(left=-1920, top=0, width=3840, height=1080)  # monitor left of primary


def test_function_key_codes() -> None:
    assert virtual_key_code("F1") == 0x70
    assert virtual_key_code("f8") == 0x77
    with pytest.raises(ValueError):
        virtual_key_code("F13")


def test_absolute_coordinates_span_the_virtual_desktop() -> None:
    assert DUAL.to_absolute(-1920, 0) == (0, 0)
    assert DUAL.to_absolute(1919, 1079) == (65535, 65535)
    ax, _ = DUAL.to_absolute(0, 0)
    assert ax == pytest.approx(65535 / 2, abs=10)


def test_recording_mouse_clamps_to_the_screen() -> None:
    mouse = RecordingMouse(DUAL, x=1900, y=10)
    mouse.move_by(50, -30)
    assert (mouse.x, mouse.y) == (1919, 0)
    mouse.move_by(0, 0)  # no-op moves are not recorded
    assert mouse.path == [(1919, 0)]


def test_input_struct_has_the_win32_size() -> None:
    # INPUT is 40 bytes on 64-bit Windows and 28 on 32-bit; a wrong size makes SendInput fail.
    expected = 40 if sys.maxsize > 2**32 else 28
    assert ctypes.sizeof(_INPUT) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_mouse_reads_a_sensible_screen() -> None:
    mouse = WindowsMouse()
    assert mouse.screen.width > 0 and mouse.screen.height > 0
    x, y = mouse.position()
    assert mouse.screen.clamp(x, y) == (x, y)


def test_hotkey_reports_one_press_per_key_down_even_when_held() -> None:
    states = iter([False, True, True, True, False, False, True, False])
    poller = HotkeyPoller.__new__(HotkeyPoller)  # skip the Win32 setup
    poller._is_down = lambda: next(states)  # type: ignore[method-assign]
    poller._was_down = False
    presses = [poller.pressed() for _ in range(8)]
    assert presses == [False, True, False, False, False, False, True, False]
