"""Standalone native capture worker. Run by ScreenCapture, never imported for discovery."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4


class Point(C.Structure):
    """Core Graphics point."""

    _fields_ = [("x", C.c_double), ("y", C.c_double)]


class Size(C.Structure):
    """Core Graphics size."""

    _fields_ = [("width", C.c_double), ("height", C.c_double)]


class Rect(C.Structure):
    """Core Graphics rectangle."""

    _fields_ = [("origin", Point), ("size", Size)]


def _function(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


def _mac_libraries():
    core = C.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    foundation = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    return core, foundation


def _mac_displays():
    core, foundation = _mac_libraries()
    colors = C.CDLL("/System/Library/Frameworks/ColorSync.framework/ColorSync")
    active = _function(
        core,
        "CGGetActiveDisplayList",
        [C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)],
        C.c_int,
    )
    bounds = _function(core, "CGDisplayBounds", [C.c_uint32], Rect)
    rotation = _function(core, "CGDisplayRotation", [C.c_uint32], C.c_double)
    make_uuid = _function(colors, "CGDisplayCreateUUIDFromDisplayID", [C.c_uint32], C.c_void_p)
    make_string = _function(foundation, "CFUUIDCreateString", [C.c_void_p, C.c_void_p], C.c_void_p)
    to_string = _function(foundation, "CFStringGetCString", [C.c_void_p, C.c_char_p, C.c_long, C.c_uint32], C.c_bool)
    release = _function(foundation, "CFRelease", [C.c_void_p], None)
    displays, count = (C.c_uint32 * 32)(), C.c_uint32(0)
    if active(32, displays, C.byref(count)):
        raise RuntimeError("Cannot enumerate macOS displays")
    result = []
    for display in displays[: count.value]:
        rect, angle = bounds(display), rotation(display)
        width, height = int(rect.size.width), int(rect.size.height)
        if angle in (90, -90, 270):
            width, height = height, width
        value, text = make_uuid(display), None
        identifier = None
        try:
            if value:
                text = make_string(None, value)
                buffer = C.create_string_buffer(256)
                if text and to_string(text, buffer, 256, 0x08000100):
                    identifier = buffer.value.decode()
        finally:
            if text:
                release(text)
            if value:
                release(value)
        result.append(
            {
                "display_id": identifier or f"unknown-{uuid4().hex}",
                "stable_id": bool(identifier),
                "bounds": [int(rect.origin.x), int(rect.origin.y), width, height],
                "rotation": angle,
            },
        )
    return result


def _mac_overlays():
    """Ignore only named floating Remy windows, preserving the normal chat window."""
    core, foundation = _mac_libraries()
    windows = _function(core, "CGWindowListCopyWindowInfo", [C.c_uint32, C.c_uint32], C.c_void_p)
    count = _function(foundation, "CFArrayGetCount", [C.c_void_p], C.c_long)
    item = _function(foundation, "CFArrayGetValueAtIndex", [C.c_void_p, C.c_long], C.c_void_p)
    get = _function(foundation, "CFDictionaryGetValue", [C.c_void_p, C.c_void_p], C.c_void_p)
    key = _function(foundation, "CFStringCreateWithCString", [C.c_void_p, C.c_char_p, C.c_uint32], C.c_void_p)
    string = _function(foundation, "CFStringGetCString", [C.c_void_p, C.c_char_p, C.c_long, C.c_uint32], C.c_bool)
    number = _function(foundation, "CFNumberGetValue", [C.c_void_p, C.c_int, C.c_void_p], C.c_bool)
    rectangle = _function(core, "CGRectMakeWithDictionaryRepresentation", [C.c_void_p, C.POINTER(Rect)], C.c_bool)
    release = _function(foundation, "CFRelease", [C.c_void_p], None)
    keys = {
        name: key(None, name.encode(), 0x08000100) for name in ("kCGWindowName", "kCGWindowLayer", "kCGWindowBounds")
    }
    array, result = windows(1, 0), []
    try:
        for index in range(count(array) if array else 0):
            window = item(array, index)
            title, layer = get(window, keys["kCGWindowName"]), get(window, keys["kCGWindowLayer"])
            text, level = C.create_string_buffer(512), C.c_long(0)
            if not title or not layer or not string(title, text, 512, 0x08000100):
                continue
            if text.value.decode() != "Remy · ReMe" or not number(layer, 4, C.byref(level)) or level.value <= 0:
                continue
            rect = Rect()
            value = get(window, keys["kCGWindowBounds"])
            if value and rectangle(value, C.byref(rect)):
                result.append([rect.origin.x, rect.origin.y, rect.size.width, rect.size.height])
    finally:
        if array:
            release(array)
        for value in keys.values():
            release(value)
    return result


def _displays(capture):
    if sys.platform == "darwin":
        return _mac_displays()
    result = []
    for monitor in capture.monitors[1:]:
        identifier = monitor.get("unique_id")
        result.append(
            {
                "display_id": str(identifier) if identifier else f"unknown-{uuid4().hex}",
                "stable_id": bool(identifier),
                "rotation": 0,
                "bounds": [monitor[name] for name in ("left", "top", "width", "height")],
            },
        )
    return result


def _overlays():
    if sys.platform == "darwin":
        try:
            return _mac_overlays()
        except (OSError, AttributeError, ValueError):
            # Missing window metadata means more analysis, never silently hiding unknown content.
            return []
    return []


def capture_displays(directory: Path, ignore_clock: bool) -> list[dict]:
    """Capture original RGB pixels; index-only identities deliberately never deduplicate."""
    import mss
    from PIL import Image

    records = []
    with mss.MSS() as capture:
        displays = _displays(capture)
        if not displays:
            raise RuntimeError("No displays available for screen capture")
        identifiers = [display["display_id"] for display in displays]
        for display in displays:
            if identifiers.count(display["display_id"]) > 1:
                display.update(display_id=f"unknown-{uuid4().hex}", stable_id=False)
            left, top, width, height = display["bounds"]
            if width <= 0 or height <= 0 or width * height > 40_000_000:
                raise ValueError("Display dimensions exceed the capture limit")
            before = _overlays()
            shot = capture.grab({"left": left, "top": top, "width": width, "height": height})
            if shot.size[0] * shot.size[1] > 40_000_000:
                raise ValueError("Screenshot dimensions exceed the capture limit")
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            sx, sy = image.width / width, image.height / height
            excluded = []
            for x, y, w, h in before + _overlays():
                x0, y0 = max(0, (x - left) * sx), max(0, (y - top) * sy)
                x1, y1 = min(image.width, (x + w - left) * sx), min(image.height, (y + h - top) * sy)
                if x1 > x0 and y1 > y0:
                    excluded.append([int(x0), int(y0), int(x1 + 0.999), int(y1 + 0.999)])
            if ignore_clock and sys.platform == "darwin":
                excluded.append([max(0, int(image.width - 250 * sx)), 0, image.width, min(image.height, int(30 * sy))])
            token = hashlib.sha256(display["display_id"].encode()).hexdigest()[:24]
            filename = f"display_{token}.png"
            image.save(directory / filename)
            records.append({**display, "path": filename, "pixel_size": list(image.size), "ignored_regions": excluded})
    return records


if __name__ == "__main__":
    print(json.dumps(capture_displays(Path(sys.argv[1]), sys.argv[2] == "1")))
