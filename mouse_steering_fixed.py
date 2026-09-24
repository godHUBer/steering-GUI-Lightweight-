#!/usr/bin/env python3
"""
mouse_steering.py -- analog steering from a mouse, through a virtual Xbox 360
controller, with a tuning GUI.

Fixed version:
- lazy-loads vgamepad so --help works without ViGEm installed;
- uses the current vgamepad VX360Gamepad lifecycle (destroying the object
  disconnects it; there is no gamepad.delete() call);
- makes the Windows process DPI-aware before pynput is used;
- makes start/stop transactional and prevents a timed-out old loop from being
  mistaken for a stopped loop;
- gives the control loop a fail-safe exception boundary;
- makes cursor-warp handling explicit and prevents synthetic warp events from
  being counted as steering input;
- validates/sanitises JSON presets;
- reports runtime failures to the GUI/CLI instead of silently leaving a false
  "running" state.

Requirements (Windows -- ViGEm is Windows-only)
------------------------------------------------
    pip install vgamepad pynput

Usage
-----
    python mouse_steering_fixed.py          # tuning GUI
    python mouse_steering_fixed.py --cli    # console-only mode
    python mouse_steering_fixed.py --help

F8  pause/resume
F9  emergency stop
Ctrl+C  quit in CLI mode

Note: pynput mouse positions are still cursor-position events rather than
Windows Raw Input HID deltas. DPI awareness is enabled so listener/controller
coordinates stay consistent, but this is not a true Raw Input implementation.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


PRESET_PATH = Path(__file__).with_name("mouse_steering.json")
CURVE_PRESETS = ["Linear", "Squared", "Cubed", "S-Curve", "Hybrid", "Custom power"]

ACCEL_V_REF = 4.0
CENTER_FLOOR = 0.15
PRECISION_BAND = 0.10
RAIL_EPS = 1e-6


# ===========================================================================
# PLATFORM / DEPENDENCIES
# ===========================================================================


def make_dpi_aware() -> None:
    """Make pynput listener/controller coordinates consistent on Windows."""
    if sys.platform != "win32":
        return

    import ctypes

    # PROCESS_PER_MONITOR_DPI_AWARE = 2.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        # If Windows is too old or the call is unavailable, leave it alone.
        # The steering code will still work on many systems, but coordinates
        # can be inconsistent under display scaling.
        pass


def import_mouse_keyboard():
    try:
        from pynput import keyboard, mouse
        return keyboard, mouse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'pynput'.\n"
            "    pip install vgamepad pynput"
        ) from exc


def import_vgamepad():
    try:
        import vgamepad
        return vgamepad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'vgamepad'.\n"
            "    pip install vgamepad pynput\n"
            "Installing vgamepad also installs the ViGEmBus driver; accept "
            "the driver installer licence."
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not initialise vgamepad/ViGEmBus.\n"
            "Make sure the ViGEmBus driver is installed and working, then "
            "retry.\n"
            f"Underlying error: {exc}"
        ) from exc


# ===========================================================================
# SETTINGS
# ===========================================================================


@dataclass
class Settings:
    # Input scaling / steering range
    full_lock_px: float = 500.0
    mouse_sensitivity: float = 1.0
    sens_left: float = 1.0
    sens_right: float = 1.0
    raw_scale: float = 1.0
    lock_left: float = 1.0
    lock_right: float = 1.0

    # Response curve
    curve_preset: str = "Cubed"
    curve_exp: float = 3.0
    precision_zone: float = 0.0
    precision_gain: float = 0.5

    # Steering dynamics
    steering_accel: float = 0.0
    max_steering_rate: float = 6.0
    saturate_clean: bool = True

    # Input filtering
    smoothing: float = 0.0
    noise_gate_px: float = 0.0
    hysteresis_px: float = 0.0

    # Centering
    center_strength: float = 1.0
    center_time_s: float = 0.30
    center_curve: float = 0.0
    idle_grace_ms: float = 60.0
    auto_return_enabled: bool = True  # spring-centre automatically when input stops

    # Output
    max_slew_rate: float = 0.0
    output_deadzone: float = 0.02
    invert_axis: bool = False

    # Timing
    update_hz: float = 83.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise ValueError("Preset root must be a JSON object.")
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        settings = cls(**values)
        return sanitise_settings(settings)


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _bool_value(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def sanitise_settings(settings: Settings) -> Settings:
    """Validate values from JSON/programmatic changes against GUI ranges."""
    s = replace(settings)

    numeric_ranges = {
        "full_lock_px": (100.0, 2000.0),
        "mouse_sensitivity": (0.05, 5.0),
        "sens_left": (0.05, 5.0),
        "sens_right": (0.05, 5.0),
        "raw_scale": (0.05, 5.0),
        "lock_left": (0.10, 1.0),
        "lock_right": (0.10, 1.0),
        "curve_exp": (0.5, 5.0),
        "precision_zone": (0.0, 0.5),
        "precision_gain": (0.05, 1.0),
        "steering_accel": (0.0, 2.0),
        "max_steering_rate": (0.5, 30.0),
        "smoothing": (0.0, 0.95),
        "noise_gate_px": (0.0, 10.0),
        "hysteresis_px": (0.0, 20.0),
        "center_strength": (0.0, 2.0),
        "center_time_s": (0.05, 2.0),
        "center_curve": (-1.0, 1.0),
        "idle_grace_ms": (0.0, 500.0),
        "max_slew_rate": (0.0, 100.0),
        "output_deadzone": (0.0, 0.2),
        "update_hz": (20.0, 500.0),
    }

    for name, (lo, hi) in numeric_ranges.items():
        value = _finite_number(name, getattr(s, name))
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}.")
        setattr(s, name, value)

    if s.curve_preset not in CURVE_PRESETS:
        raise ValueError(f"curve_preset must be one of: {', '.join(CURVE_PRESETS)}")

    s.saturate_clean = _bool_value("saturate_clean", s.saturate_clean)
    s.auto_return_enabled = _bool_value("auto_return_enabled", s.auto_return_enabled)
    s.invert_axis = _bool_value("invert_axis", s.invert_axis)
    return s


def load_preset(path: Path = PRESET_PATH) -> Settings | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = data.get("settings", data) if isinstance(data, dict) else data
        return Settings.from_dict(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_preset(settings: Settings, path: Path = PRESET_PATH) -> None:
    validated = sanitise_settings(settings)
    Path(path).write_text(
        json.dumps({"version": 1, "settings": validated.to_dict()}, indent=2),
        encoding="utf-8",
    )


# ===========================================================================
# PURE HELPERS
# ===========================================================================


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def response_curve(x: float, preset: str = "Cubed", exponent: float = 3.0) -> float:
    if x == 0.0:
        return 0.0
    a = abs(x)
    if preset == "Linear":
        r = a
    elif preset == "Squared":
        r = a * a
    elif preset == "S-Curve":
        r = a * a * (3.0 - 2.0 * a)
    elif preset == "Hybrid":
        r = 0.35 * a + 0.65 * a**3
    elif preset in ("Cubed", "Custom power"):
        r = a ** (3.0 if preset == "Cubed" else max(exponent, 0.05))
    else:
        r = a
    return math.copysign(r, x)


def centering_factor(u: float, curve: float) -> float:
    u = clamp(u, 0.0, 1.0)
    return max(CENTER_FLOOR, 1.0 + curve * (1.0 - 2.0 * u))


def precision_factor(u: float, zone: float, gain: float) -> float:
    if zone <= 0.0:
        return 1.0
    u = abs(u)
    if u <= zone:
        return gain
    if u >= zone + PRECISION_BAND:
        return 1.0
    return gain + (1.0 - gain) * smoothstep((u - zone) / PRECISION_BAND)


# ===========================================================================
# STEERING MATH
# ===========================================================================


@dataclass
class SteerState:
    pos: float = 0.0
    out: float = 0.0
    ema_dx: float = 0.0
    last_counted_t: float = 0.0
    holding: bool = False
    saturated: bool = False


def step_steering(
    st: SteerState,
    dx_px: float,
    dt: float,
    now: float,
    s: Settings,
) -> SteerState:
    ns = SteerState(
        pos=st.pos,
        out=st.out,
        ema_dx=st.ema_dx,
        last_counted_t=st.last_counted_t,
    )
    dt = max(float(dt), 1e-6)

    lock_l = max(s.lock_left, 0.01)
    lock_r = max(s.lock_right, 0.01)

    # Raw deadband first.
    if abs(dx_px) < s.noise_gate_px:
        dx_px = 0.0

    holding = (now - ns.last_counted_t) * 1000.0 < s.idle_grace_ms
    if (not holding) and abs(dx_px) < s.hysteresis_px:
        dx_px = 0.0
    phys = dx_px != 0.0

    # EMA.
    a = clamp(s.smoothing, 0.0, 0.95)
    if a > 0.0:
        ns.ema_dx = a * ns.ema_dx + (1.0 - a) * dx_px
        dx_f = ns.ema_dx
    else:
        ns.ema_dx = 0.0
        dx_f = dx_px

    # Apply the deadband again after filtering so the configured noise gate
    # remains a true lower bound on counted motion.
    if abs(dx_f) < s.noise_gate_px:
        dx_f = 0.0
    if (not holding) and abs(dx_f) < s.hysteresis_px:
        dx_f = 0.0
        ns.ema_dx = 0.0

    du = 0.0
    if dx_f:
        lock_side = lock_r if dx_f > 0.0 else lock_l
        u = abs(ns.pos) / lock_side
        sens = s.sens_right if dx_f > 0.0 else s.sens_left
        dx_eff = (
            dx_f
            * s.raw_scale
            * s.mouse_sensitivity
            * sens
            * precision_factor(u, s.precision_zone, s.precision_gain)
        )
        du = dx_eff / max(s.full_lock_px, 1.0)

        if s.steering_accel > 0.0:
            flick = min(abs(du) / dt / ACCEL_V_REF, 1.0)
            du *= 1.0 + s.steering_accel * flick

        if s.max_steering_rate > 0.0:
            du = clamp(du, -s.max_steering_rate * dt, s.max_steering_rate * dt)

    sat = False
    skip_decay = False
    if du > 0.0 and ns.pos >= lock_r - RAIL_EPS:
        sat = True
    elif du < 0.0 and ns.pos <= -lock_l + RAIL_EPS:
        sat = True

    if sat and s.saturate_clean:
        du = 0.0
        if phys:
            skip_decay = True
        else:
            ns.ema_dx = 0.0

    if du:
        ns.pos = clamp(ns.pos + du, -lock_l, lock_r)
        ns.last_counted_t = now
        holding = True

    if (
        (not holding)
        and (not skip_decay)
        and ns.pos != 0.0
        and s.auto_return_enabled
        and s.center_strength > 0.0
        and s.center_time_s > 0.0
    ):
        side = lock_r if ns.pos > 0.0 else lock_l
        u = clamp(abs(ns.pos) / side, 0.0, 1.0)
        rate = (
            s.center_strength
            / s.center_time_s
            * centering_factor(u, s.center_curve)
        )
        step = rate * side * dt
        if ns.pos > 0.0:
            ns.pos = max(0.0, ns.pos - step)
        else:
            ns.pos = min(0.0, ns.pos + step)

    ns.holding = holding
    ns.saturated = sat

    side = lock_r if ns.pos >= 0.0 else lock_l
    u = clamp(ns.pos / side, -1.0, 1.0) if side > 0.0 else 0.0
    x = response_curve(u, s.curve_preset, s.curve_exp) * side

    if s.max_slew_rate > 0.0:
        x = ns.out + clamp(x - ns.out, -s.max_slew_rate * dt, s.max_slew_rate * dt)

    ns.out = clamp(x)
    return ns


def to_axis(out: float, s: Settings) -> float:
    if out == 0.0:
        return 0.0
    x = out
    if abs(x) < s.output_deadzone:
        x = 0.0
    if s.invert_axis:
        x = -x
    return clamp(x)


def travel_px(s: Settings) -> tuple[float, float]:
    raw = max(s.raw_scale * s.mouse_sensitivity, 1e-9)
    left = (
        max(s.full_lock_px, 1.0)
        * s.lock_left
        / (raw * max(s.sens_left, 1e-9))
    )
    right = (
        max(s.full_lock_px, 1.0)
        * s.lock_right
        / (raw * max(s.sens_right, 1e-9))
    )
    return left, right


def format_status(steer: float, axis: float, paused: bool) -> str:
    width = 11
    cells = ["-"] * (2 * width + 1)
    cells[width] = "|"
    idx = width + int(round(clamp(axis) * width))
    if idx != width:
        cells[max(0, min(idx, 2 * width))] = "#"
    state = "paused" if paused else "active"
    return f"\r[{''.join(cells)}] {axis:+6.1%}  steer={steer:+.3f}  ({state})   "


# ===========================================================================
# SCREEN / MOUSE
# ===========================================================================


def get_screen_center() -> tuple[int, int] | None:
    """Centre of the desktop, spanning all monitors, when available."""
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        try:
            x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            y = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            w = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            h = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            return (x + w // 2, y + h // 2)
        except Exception:
            return (
                user32.GetSystemMetrics(0) // 2,
                user32.GetSystemMetrics(1) // 2,
            )

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        center = (root.winfo_screenwidth() // 2, root.winfo_screenheight() // 2)
        root.destroy()
        return center
    except Exception:
        return None


class RelativeMouseTracker:
    """
    Accumulates horizontal cursor movement.

    Cursor recentering is explicit: the control loop asks the tracker to warp
    the cursor, and the next mouse move event is consumed as the synthetic
    rebase event rather than steering input. Windows DPI awareness is applied
    before this object is created, avoiding listener/controller coordinate
    mismatches caused by display scaling.
    """

    def __init__(self, mouse_module, recenter: bool = True) -> None:
        self._lock = threading.Lock()
        self._accum_dx = 0.0
        self._last_xy: tuple[float, float] | None = None
        self._ignore_next_move = False
        self._paused = False
        self._controller = mouse_module.Controller()
        self._screen_center = get_screen_center() if recenter else None

    def on_move(self, x: float, y: float) -> None:
        with self._lock:
            # A cursor warp generates a mouse move event. Consume one such
            # event as a rebase. Do not compare exact coordinates: doing so
            # is fragile when another event is delivered first.
            if self._ignore_next_move:
                self._ignore_next_move = False
                self._last_xy = (x, y)
                return

            if self._last_xy is not None and not self._paused:
                dx = x - self._last_xy[0]
                if dx:
                    self._accum_dx += dx
            self._last_xy = (x, y)

    def take_dx(self) -> float:
        with self._lock:
            dx = self._accum_dx
            self._accum_dx = 0.0
            return dx

    def warp_to_center(self) -> None:
        center = self._screen_center
        if center is None:
            return
        with self._lock:
            # Clear already accumulated input before the synthetic move. The
            # following callback is consumed as the rebase event.
            self._accum_dx = 0.0
            self._ignore_next_move = True
            self._last_xy = center
        try:
            self._controller.position = center
        except Exception:
            # If warping fails, drop the synthetic-ignore state so the tracker
            # does not permanently discard a future real mouse event.
            with self._lock:
                self._ignore_next_move = False

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def recentering(self) -> bool:
        return self._screen_center is not None

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
            self._accum_dx = 0.0
            # Keep last_xy tracking across a pause. The next physical move is
            # therefore measured relative to the current cursor position.


# ===========================================================================
# GAMEPAD
# ===========================================================================


def create_gamepad():
    vg = import_vgamepad()
    cls = getattr(vg, "VX360Gamepad", None)
    if cls is None:
        raise RuntimeError(
            "This vgamepad build does not expose VX360Gamepad."
        )
    try:
        return cls()
    except Exception as exc:
        raise RuntimeError(
            "Could not create the virtual Xbox 360 gamepad.\n"
            "Is the ViGEmBus driver installed and working?\n"
            f"Underlying error: {exc}"
        ) from exc


def shutdown_gamepad(gamepad) -> None:
    """
    Neutralise and reset the virtual pad.

    Current vgamepad keeps the device connected for the lifetime of the
    VX360Gamepad object and disconnects when the object is destroyed, so this
    function deliberately does NOT call a nonexistent delete() method.
    """
    if gamepad is None:
        return
    try:
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.update()
        gamepad.reset()
        gamepad.update()
    except Exception:
        # The object will still be released by the owner dropping its reference.
        pass


# ===========================================================================
# STEERING ENGINE
# ===========================================================================


class SteeringEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        recenter: bool = True,
        show_status: bool = False,
    ) -> None:
        self._settings = sanitise_settings(settings or Settings())
        self._slock = threading.Lock()
        self._tlock = threading.Lock()
        self._recenter = recenter
        self._show_status = show_status

        self._gamepad = None
        self._tracker: RelativeMouseTracker | None = None
        self._mouse_listener = None
        self._hotkeys = None
        self._keyboard = None
        self._mouse = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ops = threading.Lock()
        self._telemetry = {
            "pos": 0.0,
            "out": 0.0,
            "holding": False,
            "saturated": False,
            "running": False,
            "paused": False,
            "error": None,
        }

    # ----- settings ----------------------------------------------------
    def get_settings(self) -> Settings:
        with self._slock:
            return replace(self._settings)

    def set_field(self, key: str, value) -> None:
        with self._slock:
            if not hasattr(self._settings, key):
                raise AttributeError(f"Unknown setting: {key}")
            candidate = replace(self._settings)
            setattr(candidate, key, value)
            self._settings = sanitise_settings(candidate)

    def replace_settings(self, settings: Settings) -> None:
        validated = sanitise_settings(settings)
        with self._slock:
            self._settings = replace(validated)

    # ----- telemetry ---------------------------------------------------
    def telemetry(self) -> dict[str, Any]:
        with self._tlock:
            return dict(self._telemetry)

    def _set_telemetry(self, **kw) -> None:
        with self._tlock:
            self._telemetry.update(kw)

    # ----- lifecycle ---------------------------------------------------
    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _stop_listeners(self) -> None:
        for listener in (self._mouse_listener, self._hotkeys):
            try:
                if listener is not None:
                    listener.stop()
            except Exception:
                pass
        self._mouse_listener = None
        self._hotkeys = None
        self._keyboard = None
        self._mouse = None
        self._tracker = None

    def _cleanup_stopped_runtime(self) -> None:
        """Cleanup resources only after the control thread is confirmed dead."""
        self._stop_listeners()
        gamepad = self._gamepad
        self._gamepad = None
        shutdown_gamepad(gamepad)

    def start(self) -> None:
        with self._ops:
            if self.running:
                return

            # If a previous loop died unexpectedly, remove its leftover
            # listeners/gamepad before starting a new session.
            self._cleanup_stopped_runtime()

            make_dpi_aware()
            self._keyboard, self._mouse = import_mouse_keyboard()

            gamepad = None
            tracker = None
            mouse_listener = None
            hotkeys = None
            try:
                gamepad = create_gamepad()
                tracker = RelativeMouseTracker(self._mouse, recenter=self._recenter)

                self._gamepad = gamepad
                self._tracker = tracker
                self._stop_event = threading.Event()
                self._set_telemetry(
                    running=True,
                    paused=False,
                    pos=0.0,
                    out=0.0,
                    holding=False,
                    saturated=False,
                    error=None,
                    auto_return_enabled=self.get_settings().auto_return_enabled,
                )

                def _on_pause() -> None:
                    t = self.telemetry()
                    self.set_paused(not bool(t["paused"]))

                def _on_stop() -> None:
                    # Don't join/stop from the callback thread itself.
                    threading.Thread(
                        target=self.stop,
                        daemon=True,
                        name="steering-emergency-stop",
                    ).start()

                mouse_listener = self._mouse.Listener(on_move=tracker.on_move)
                mouse_listener.start()

                hotkeys = self._keyboard.GlobalHotKeys(
                    {"<f8>": _on_pause, "<f9>": _on_stop}
                )
                hotkeys.start()

                thread = threading.Thread(
                    target=self._loop,
                    daemon=True,
                    name="steering-control-loop",
                )
                self._mouse_listener = mouse_listener
                self._hotkeys = hotkeys
                self._thread = thread
                thread.start()

            except Exception:
                # Transactional startup: no half-started engine survives.
                try:
                    if mouse_listener is not None:
                        mouse_listener.stop()
                except Exception:
                    pass
                try:
                    if hotkeys is not None:
                        hotkeys.stop()
                except Exception:
                    pass

                self._mouse_listener = None
                self._hotkeys = None
                self._tracker = None
                self._keyboard = None
                self._mouse = None
                self._thread = None
                self._gamepad = None
                shutdown_gamepad(gamepad)
                self._set_telemetry(
                    running=False,
                    paused=False,
                    error="Could not start steering engine.",
                )
                raise

    def stop(self, timeout: float = 2.0) -> None:
        with self._ops:
            self._stop_event.set()
            thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout)

            # Never claim a still-running thread has stopped. This prevents a
            # second control loop from being started on top of the first.
            if thread is not None and thread.is_alive():
                self._set_telemetry(
                    running=True,
                    error="Control loop did not stop within the timeout.",
                )
                return

            self._thread = None
            self._cleanup_stopped_runtime()
            self._set_telemetry(
                running=False,
                paused=False,
                pos=0.0,
                out=0.0,
                holding=False,
                saturated=False,
            )

    def set_paused(self, paused: bool) -> None:
        tracker = self._tracker
        if tracker is not None:
            tracker.set_paused(paused)
        self._set_telemetry(paused=paused)

    def wait(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join()

    # ----- loop --------------------------------------------------------
    def _loop(self) -> None:
        state = SteerState(last_counted_t=time.perf_counter())
        next_t = time.perf_counter()
        last_t = next_t
        last_status = 0.0
        fault: str | None = None

        try:
            while not self._stop_event.is_set():
                now = time.perf_counter()
                dt = max(now - last_t, 1e-6)
                last_t = now

                s = self.get_settings()
                tracker = self._tracker
                gamepad = self._gamepad
                paused = tracker is not None and tracker.paused
                dx = tracker.take_dx() if tracker is not None else 0.0

                if paused:
                    state = SteerState()
                    axis = 0.0
                    if gamepad is not None:
                        gamepad.left_joystick_float(
                            x_value_float=0.0,
                            y_value_float=0.0,
                        )
                        gamepad.update()
                else:
                    state = step_steering(state, dx, dt, now, s)
                    axis = to_axis(state.out, s)
                    if gamepad is not None:
                        gamepad.left_joystick_float(
                            x_value_float=axis,
                            y_value_float=0.0,
                        )
                        gamepad.update()

                if tracker is not None and self._recenter and not paused:
                    tracker.warp_to_center()

                self._set_telemetry(
                    pos=state.pos,
                    out=axis,
                    holding=state.holding,
                    saturated=state.saturated,
                    paused=paused,
                    running=True,
                    auto_return_enabled=s.auto_return_enabled,
                )

                if (
                    self._show_status
                    and hasattr(sys.stdout, "isatty")
                    and sys.stdout.isatty()
                    and now - last_status >= 0.2
                ):
                    print(format_status(state.pos, axis, paused), end="", flush=True)
                    last_status = now

                hz = s.update_hz if s.update_hz > 0.0 else 83.0
                tick = 1.0 / min(max(hz, 1.0), 1000.0)
                next_t += tick
                delay = next_t - time.perf_counter()
                if delay > 0.0:
                    # Wake promptly on stop instead of sleeping a full tick.
                    self._stop_event.wait(delay)
                else:
                    next_t = time.perf_counter()

        except Exception as exc:
            fault = f"Steering loop stopped: {exc}"
            self._set_telemetry(
                running=False,
                paused=False,
                error=fault,
            )
        finally:
            # Fail-safe neutralisation even if update() or another loop
            # operation raises unexpectedly.
            try:
                if self._gamepad is not None:
                    shutdown_gamepad(self._gamepad)
            except Exception:
                pass

            self._stop_event.set()
            if fault is None:
                self._set_telemetry(running=False)


# ===========================================================================
# CLI
# ===========================================================================


def run_cli(settings: Settings) -> None:
    l, r = travel_px(settings)
    print("=" * 62)
    print("  Mouse -> Analog Steering  (virtual Xbox 360 controller)")
    print("=" * 62)
    print(
        f"  steering range : {settings.full_lock_px:.0f} px = 100% "
        f"(travel L≈{l:.0f} / R≈{r:.0f} px)"
    )
    print(
        f"  sensitivity    : {settings.mouse_sensitivity:.2f} "
        f"(L {settings.sens_left:.2f} / R {settings.sens_right:.2f})"
    )
    curve_extra = (
        f" (exp {settings.curve_exp:g})"
        if settings.curve_preset == "Custom power"
        else ""
    )
    print(f"  curve          : {settings.curve_preset}{curve_extra}")
    print(
        f"  centering      : strength {settings.center_strength:.2f}, "
        f"{settings.center_time_s * 1000:.0f} ms, curve {settings.center_curve:+.2f}"
    )
    print(f"  update rate    : {settings.update_hz:.0f} Hz")
    print("-" * 62)
    print("  Move mouse left/right to steer; it self-centres when you stop.")
    print("  F8 = pause/resume   F9 = emergency stop   Ctrl+C = quit")
    print("=" * 62, flush=True)

    engine = SteeringEngine(settings, show_status=True)
    try:
        engine.start()
    except RuntimeError as exc:
        sys.exit(str(exc))

    print("Virtual Xbox 360 pad connected.\n", flush=True)
    try:
        engine.wait()
        if sys.stdout.isatty():
            print()
    except KeyboardInterrupt:
        print("\nCtrl+C -- shutting down...")
    finally:
        engine.stop()
        telemetry = engine.telemetry()
        if telemetry.get("error"):
            print(f"Engine status: {telemetry['error']}")
        print("Virtual pad returned to neutral and disconnected. Bye.", flush=True)


# ===========================================================================
# GUI
# ===========================================================================


ADV_GROUPS: list[tuple[str, list]] = [
    (
        "Input scaling",
        [
            (
                "raw_scale",
                "Raw-input scale",
                0.05,
                5.0,
                0.05,
                "DPI / polling normalisation (0.5 = 1600-DPI mouse tuned like 800)",
                "%.2f",
            ),
            (
                "sens_left",
                "Sensitivity LEFT",
                0.05,
                5.0,
                0.05,
                "Steering gain when moving left",
                "%.2f",
            ),
            (
                "sens_right",
                "Sensitivity RIGHT",
                0.05,
                5.0,
                0.05,
                "Steering gain when moving right",
                "%.2f",
            ),
            (
                "lock_left",
                "Max lock LEFT",
                0.10,
                1.0,
                0.01,
                "How far toward full-left the stick may reach",
                "%.2f",
            ),
            (
                "lock_right",
                "Max lock RIGHT",
                0.10,
                1.0,
                0.01,
                "How far toward full-right the stick may reach",
                "%.2f",
            ),
        ],
    ),
    (
        "Input filtering",
        [
            (
                "smoothing",
                "Input smoothing (EMA)",
                0.0,
                0.95,
                0.05,
                "Smooths tiny jitter from mouse movement",
                "%.2f",
            ),
            (
                "noise_gate_px",
                "Noise gate (px/tick)",
                0.0,
                10.0,
                0.25,
                "Ignore per-tick movement below this",
                "%.2f",
            ),
            (
                "hysteresis_px",
                "Centering hysteresis (px/tick)",
                0.0,
                20.0,
                0.5,
                "While centring, trembles below this cannot fight the spring",
                "%.2f",
            ),
            (
                "precision_zone",
                "Low-speed precision zone",
                0.0,
                0.5,
                0.01,
                "Near-centre region where movement is scaled down",
                "%.2f",
            ),
            (
                "precision_gain",
                "Precision zone gain",
                0.05,
                1.0,
                0.05,
                "Movement scale inside the precision zone",
                "%.2f",
            ),
        ],
    ),
    (
        "Steering dynamics",
        [
            (
                "steering_accel",
                "Steering acceleration",
                0.0,
                2.0,
                0.05,
                "Fast movement builds steering more quickly",
                "%.2f",
            ),
            (
                "max_steering_rate",
                "Max steering rate (locks/s)",
                0.5,
                30.0,
                0.5,
                "A flick cannot reach full lock instantly",
                "%.1f",
            ),
            (
                "bool",
                "saturate_clean",
                "Clean saturation",
                "Movement pushing into an engaged lock leaves no trace",
            ),
        ],
    ),
    (
        "Centering",
        [
            (
                "idle_grace_ms",
                "Hold time before centering (ms)",
                0.0,
                500.0,
                10.0,
                "Short hand pauses keep the line before the spring starts",
                "%.0f",
            ),
        ],
    ),
    (
        "Output",
        [
            (
                "max_slew_rate",
                "Max output slew (sticks/s)",
                0.0,
                100.0,
                1.0,
                "Limits how fast the virtual stick may change (0 = unlimited)",
                "%.0f",
            ),
            (
                "output_deadzone",
                "Output deadzone",
                0.0,
                0.2,
                0.005,
                "Values smaller than this are sent as 0",
                "%.3f",
            ),
            (
                "bool",
                "invert_axis",
                "Invert axis",
                "Reverse if the game turns the wrong way",
            ),
        ],
    ),
    (
        "Timing",
        [
            (
                "update_hz",
                "Update frequency (Hz)",
                20.0,
                500.0,
                1.0,
                "Virtual gamepad report rate",
                "%.0f",
            ),
        ],
    ),
]


def run_gui(settings: Settings) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:  # pragma: no cover
        sys.exit("tkinter is not available -- install it or use --cli mode.")

    engine = SteeringEngine(settings, show_status=False)

    class SteeringApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Mouse -> Analog Steering")
            self.resizable(False, False)
            self.engine = engine
            self.vars: dict[str, tuple[Any, str | None]] = {}
            self.labels: dict[str, Any] = {}
            self._in_update = False

            self._auto_return_var = tk.BooleanVar(
                value=self.engine.get_settings().auto_return_enabled
            )

            self._build_header()
            self._build_meter()
            self._build_notebook()
            self._build_buttons()
            self._build_statusbar()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.after(50, self._poll)

        def set_field(self, key: str, value) -> None:
            try:
                self.engine.set_field(key, value)
            except (AttributeError, ValueError) as exc:
                self.status.config(text=str(exc))

        def slider_row(
            self,
            parent,
            key,
            label,
            lo,
            hi,
            res,
            hint,
            fmt,
        ) -> None:
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=1)
            ttk.Label(row, text=label, width=26).pack(side="left")
            var = tk.DoubleVar(value=getattr(self.engine.get_settings(), key))
            self.vars[key] = (var, fmt)

            def on_move(v, key=key, var=var, res=res, fmt=fmt):
                if self._in_update:
                    return
                val = float(v)
                if res:
                    val = round(val / res) * res
                val = max(lo, min(hi, val))
                self._in_update = True
                try:
                    var.set(val)
                finally:
                    self._in_update = False
                self.set_field(key, val)
                self.labels[key].config(text=fmt % val)

            scale = ttk.Scale(
                row,
                from_=lo,
                to=hi,
                variable=var,
                length=180,
                command=on_move,
            )
            scale.pack(side="left", padx=4)
            lbl = ttk.Label(row, text=fmt % var.get(), width=7, anchor="e")
            lbl.pack(side="left")
            self.labels[key] = lbl
            if hint:
                ttk.Label(
                    row,
                    text=hint,
                    foreground="#666",
                    font=("TkDefaultFont", 8),
                ).pack(side="left", padx=6)

        def bool_row(self, parent, key, label, hint) -> None:
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=8, pady=1)
            var = tk.BooleanVar(value=getattr(self.engine.get_settings(), key))
            self.vars[key] = (var, None)
            ttk.Checkbutton(
                row,
                text=label,
                variable=var,
                command=lambda: self.set_field(key, var.get()),
            ).pack(side="left")
            if hint:
                ttk.Label(
                    row,
                    text=hint,
                    foreground="#666",
                    font=("TkDefaultFont", 8),
                ).pack(side="left", padx=6)

        def _build_header(self) -> None:
            bar = ttk.Frame(self, padding=(10, 8))
            bar.pack(fill="x")
            self.start_btn = ttk.Button(bar, text="Start", command=self._toggle_run)
            self.start_btn.pack(side="left")
            self.pause_btn = ttk.Button(
                bar,
                text="Pause",
                command=self._toggle_pause,
                state="disabled",
            )
            self.pause_btn.pack(side="left", padx=6)

            self.auto_return_btn = ttk.Button(
                bar,
                text="Auto return: ON",
                command=self._toggle_auto_return,
            )
            self.auto_return_btn.pack(side="left", padx=6)

            self.state_lbl = ttk.Label(bar, text="Stopped", foreground="#a33")
            self.state_lbl.pack(side="left", padx=12)
            ttk.Label(
                bar,
                text="F8 pause/resume   F9 emergency stop",
                foreground="#666",
            ).pack(side="right")

        def _build_meter(self) -> None:
            frame = ttk.LabelFrame(self, text="Steering", padding=6)
            frame.pack(fill="x", padx=10, pady=4)
            self.canvas = tk.Canvas(
                frame,
                width=760,
                height=86,
                highlightthickness=0,
                bg="#1e1e1e",
            )
            self.canvas.pack()
            self.meter_lbl = ttk.Label(frame, text="steer +0.000   out +0.0%")
            self.meter_lbl.pack(anchor="e")

        def _build_notebook(self) -> None:
            nb = ttk.Notebook(self)
            nb.pack(fill="both", expand=True, padx=10, pady=4)

            main = ttk.Frame(nb, padding=6)
            nb.add(main, text="  Main  ")

            self.slider_row(
                main,
                "full_lock_px",
                "Steering range (px = 100%)",
                100.0,
                2000.0,
                10.0,
                "Distance for a full turn",
                "%.0f",
            )
            self.slider_row(
                main,
                "mouse_sensitivity",
                "Mouse sensitivity",
                0.05,
                5.0,
                0.05,
                "Independent of the steering range above",
                "%.2f",
            )

            row = ttk.Frame(main)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text="Response-curve preset", width=26).pack(side="left")
            self.curve_var = tk.StringVar(
                value=self.engine.get_settings().curve_preset
            )
            combo = ttk.Combobox(
                row,
                textvariable=self.curve_var,
                values=CURVE_PRESETS,
                state="readonly",
                width=14,
            )
            combo.pack(side="left", padx=4)
            combo.bind("<<ComboboxSelected>>", self._on_curve_preset)

            self.slider_row(
                main,
                "curve_exp",
                "Curve exponent (Custom power)",
                0.5,
                5.0,
                0.1,
                "",
                "%.1f",
            )
            self.slider_row(
                main,
                "center_strength",
                "Centering strength",
                0.0,
                2.0,
                0.05,
                "Spring power (0 = wheel stays where you leave it)",
                "%.2f",
            )
            self.slider_row(
                main,
                "center_time_s",
                "Center-return time (s)",
                0.05,
                2.0,
                0.05,
                "Approximate internal steering return time",
                "%.2f",
            )
            self.slider_row(
                main,
                "center_curve",
                "Centering curve",
                -1.0,
                1.0,
                0.05,
                "strong at lock <-> linear <-> strong at centre",
                "%+.2f",
            )

            self.travel_lbl = ttk.Label(main, text="", foreground="#444")
            self.travel_lbl.pack(anchor="w", padx=8, pady=4)
            ttk.Label(
                main,
                text="Auto return can be toggled at the top: OFF keeps the wheel where you leave it.",
                foreground="#666",
                font=("TkDefaultFont", 8),
            ).pack(anchor="w", padx=8, pady=(0, 4))

            adv = ttk.Frame(nb, padding=4)
            nb.add(adv, text="  Advanced settings  ")
            for group_name, items in ADV_GROUPS:
                group = ttk.LabelFrame(adv, text=group_name, padding=4)
                group.pack(fill="x", padx=6, pady=3)
                for item in items:
                    if item[0] == "bool":
                        _, key, label, hint = item
                        self.bool_row(group, key, label, hint)
                    else:
                        self.slider_row(group, *item)

        def _build_buttons(self) -> None:
            bar = ttk.Frame(self, padding=(10, 4))
            bar.pack(fill="x")
            ttk.Button(
                bar,
                text="Save preset…",
                command=self._save_preset,
            ).pack(side="left")
            ttk.Button(
                bar,
                text="Load preset…",
                command=self._load_preset,
            ).pack(side="left", padx=6)
            ttk.Button(
                bar,
                text="Defaults",
                command=self._reset_defaults,
            ).pack(side="left")

        def _build_statusbar(self) -> None:
            self.status = ttk.Label(
                self,
                text="Ready.",
                relief="sunken",
                anchor="w",
                padding=(6, 2),
            )
            self.status.pack(fill="x", side="bottom")

        def _toggle_run(self) -> None:
            if self.engine.running:
                self.engine.stop()
            else:
                try:
                    self.engine.start()
                    self.status.config(text="Started.")
                except RuntimeError as exc:
                    self.status.config(text=str(exc).splitlines()[0])
            self._sync_buttons()

        def _toggle_pause(self) -> None:
            if not self.engine.running:
                return
            t = self.engine.telemetry()
            self.engine.set_paused(not bool(t["paused"]))
            self._sync_buttons()

        def _toggle_auto_return(self) -> None:
            enabled = not bool(self.engine.get_settings().auto_return_enabled)
            self.set_field("auto_return_enabled", enabled)
            self._auto_return_var.set(enabled)
            self.auto_return_btn.config(
                text="Auto return: ON" if enabled else "Auto return: OFF"
            )
            self.status.config(
                text=(
                    "Auto return enabled. The wheel will spring back when input stops."
                    if enabled
                    else "Auto return disabled. The wheel stays where you leave it."
                )
            )

        def _sync_buttons(self) -> None:
            t = self.engine.telemetry()
            running = bool(t["running"])
            paused = bool(t["paused"])
            error = t.get("error")

            auto_return = bool(self.engine.get_settings().auto_return_enabled)
            self._auto_return_var.set(auto_return)
            self.auto_return_btn.config(
                text="Auto return: ON" if auto_return else "Auto return: OFF"
            )

            self.start_btn.config(text="Stop" if running else "Start")
            self.pause_btn.config(
                text="Resume" if paused else "Pause",
                state="normal" if running else "disabled",
            )

            if not running:
                if error:
                    self.state_lbl.config(text="Error", foreground="#a33")
                else:
                    self.state_lbl.config(text="Stopped", foreground="#a33")
            elif paused:
                self.state_lbl.config(text="Paused", foreground="#a60")
            else:
                self.state_lbl.config(text="Running", foreground="#2a2")

        def _on_curve_preset(self, _event=None) -> None:
            preset = self.curve_var.get()
            self.set_field("curve_preset", preset)
            implied = {"Squared": 2.0, "Cubed": 3.0}
            if preset in implied:
                self.set_field("curve_exp", implied[preset])
                var, fmt = self.vars["curve_exp"]
                var.set(implied[preset])
                self.labels["curve_exp"].config(text=fmt % implied[preset])

        def _save_preset(self) -> None:
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=PRESET_PATH.name,
                filetypes=[("JSON preset", "*.json")],
            )
            if path:
                try:
                    save_preset(self.engine.get_settings(), Path(path))
                    self.status.config(text=f"Saved {path}")
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Save preset", str(exc))

        def _load_preset(self) -> None:
            path = filedialog.askopenfilename(
                filetypes=[("JSON preset", "*.json")]
            )
            if not path:
                return
            try:
                loaded = load_preset(Path(path))
                if loaded is None:
                    raise ValueError("Invalid or unreadable preset.")
                self.engine.replace_settings(loaded)
                self._refresh_widgets(loaded)
                self.status.config(text=f"Loaded {path}")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Load preset", str(exc))

        def _reset_defaults(self) -> None:
            defaults = Settings()
            self.engine.replace_settings(defaults)
            self._refresh_widgets(defaults)
            self.status.config(text="All settings reset to defaults.")

        def _refresh_widgets(self, s: Settings) -> None:
            self._in_update = True
            try:
                for key, (var, fmt) in self.vars.items():
                    val = getattr(s, key)
                    var.set(val)
                    if fmt is not None and key in self.labels:
                        self.labels[key].config(text=fmt % val)
                self.curve_var.set(s.curve_preset)
                self._auto_return_var.set(s.auto_return_enabled)
                self.auto_return_btn.config(
                    text="Auto return: ON" if s.auto_return_enabled else "Auto return: OFF"
                )
            finally:
                self._in_update = False

        def _poll(self) -> None:
            try:
                t = self.engine.telemetry()
                s = self.engine.get_settings()
                c = self.canvas
                c.delete("all")
                w, h, y0 = 760, 86, 46
                cx = w // 2
                half = w // 2 - 30

                c.create_line(30, y0, w - 30, y0, fill="#444")
                c.create_line(cx, y0 - 18, cx, y0 + 18, fill="#888")

                lock_r = max(s.lock_right, 0.01)
                lock_l = max(s.lock_left, 0.01)
                xr = cx + half * lock_r
                xl = cx - half * lock_l
                c.create_line(xr, y0 - 12, xr, y0 + 12, fill="#a55")
                c.create_line(xl, y0 - 12, xl, y0 + 12, fill="#a55")
                c.create_text(xr + 14, y0 + 22, text="R lock", fill="#a55")
                c.create_text(xl - 14, y0 + 22, text="L lock", fill="#a55")

                pos = float(t["pos"])
                out = float(t["out"])
                x_pos = cx + half * pos
                x_out = cx + half * out
                c.create_line(cx, y0, x_pos, y0, fill="#2a7", width=6)
                c.create_line(x_pos, y0 - 14, x_pos, y0 + 14, fill="#5f5", width=2)
                c.create_oval(
                    x_out - 5,
                    y0 - 5,
                    x_out + 5,
                    y0 + 5,
                    outline="#8cf",
                    fill="#26a",
                )

                state = (
                    "PAUSED"
                    if t["paused"]
                    else ("running" if t["running"] else "stopped")
                )
                extra = (
                    "  [shoving at lock]"
                    if t["saturated"]
                    else ("  [holding]" if t["holding"] else "")
                )
                c.create_text(
                    30,
                    16,
                    anchor="w",
                    fill="#aaa",
                    text=f"{state}{extra}",
                )
                self.meter_lbl.config(
                    text=f"steer {pos:+.3f}   out {out:+.1%}"
                )

                l, r = travel_px(s)
                self.travel_lbl.config(
                    text=f"Travel to full lock:  L ≈ {l:.0f} px   R ≈ {r:.0f} px"
                )

                if t.get("error") and not t["running"]:
                    self.status.config(text=str(t["error"]))

                self._sync_buttons()
            finally:
                if self.winfo_exists():
                    self.after(50, self._poll)

        def _on_close(self) -> None:
            try:
                self.engine.stop()
            finally:
                self.destroy()

    app = SteeringApp()
    app.mainloop()


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> None:
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    if sys.platform != "win32":
        print(
            "Warning: ViGEm is Windows-only; this program is intended for Windows.",
            file=sys.stderr,
        )

    try:
        settings = load_preset() or Settings()
        settings = sanitise_settings(settings)
        if "--cli" in args:
            run_cli(settings)
        else:
            run_gui(settings)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
