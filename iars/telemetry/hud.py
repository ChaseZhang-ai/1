"""Head-up display compositor.

Turns a fused navigation state into the symbology the wearer actually sees:
speed and altitude tapes, a heading ribbon, vertical-speed indicator, the
"hard-deck" and pull-altitude warnings, and the AAD status line.  The reference
flight profile (freefall, canopy, smoke diving) selects the active symbology
set.  Everything rendered here is English-only so the exported frames can be
published.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/Library/Fonts/Arial.ttf",
]

GREEN = (60, 255, 120)
RED = (255, 90, 90)
AMBER = (255, 190, 60)
WHITE = (230, 255, 240)


def _font(size: int) -> ImageFont.ImageFont:
    for path in MONO_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class HudState:
    speed_knots: float = 115.0
    speed_gs_knots: float = 102.0
    altitude_ft: float = 3710.0
    hard_deck_ft: float = 2500.0
    pull_ft: float = 3200.0
    vertical_speed_mph: float = 118.0
    heading_deg: float = 313.0
    g_load: float = 1.0
    aad_armed: bool = True
    profile: str = "FREEFALL"


def render_hud(state: HudState | None = None, size: tuple = (1600, 900)) -> Image.Image:
    state = state or HudState()
    w, h = size
    img = Image.new("RGB", size, (4, 10, 8))
    d = ImageDraw.Draw(img)

    f_small = _font(20)
    f_med = _font(26)
    f_big = _font(64)
    f_warn = _font(40)
    f_top = _font(34)

    # Slim frame corners.
    for (cx, cy, sx, sy) in [(24, 24, 1, 1), (w - 24, 24, -1, 1), (24, h - 24, 1, -1), (w - 24, h - 24, -1, -1)]:
        d.line([(cx, cy), (cx + sx * 70, cy)], fill=GREEN, width=2)
        d.line([(cx, cy), (cx, cy + sy * 70)], fill=GREEN, width=2)

    # ---- left speed tape -------------------------------------------------
    lx = 150
    d.line([(lx, 140), (lx, 760)], fill=GREEN, width=2)
    d.text((lx - 70, 104), "Knots", font=f_med, fill=GREEN)
    for k in range(0, 9):
        val = int(state.speed_knots // 10 * 10 + (4 - k) * 5)
        y = 160 + k * 70
        d.line([(lx - 10, y), (lx, y)], fill=GREEN, width=2)
        d.text((lx - 18, y - 12), f"{val}", font=f_small, fill=GREEN, anchor="ra")

    # ---- right altitude tape --------------------------------------------
    rx = w - 150
    d.line([(rx, 140), (rx, 760)], fill=GREEN, width=2)
    d.text((rx + 20, 104), "Feet", font=f_med, fill=GREEN)
    for k in range(0, 9):
        val = int(state.altitude_ft // 500 * 500 + (4 - k) * 500)
        y = 160 + k * 70
        d.line([(rx, y), (rx + 10, y)], fill=GREEN, width=2)
        d.text((rx + 18, y - 12), f"{val}", font=f_small, fill=GREEN)

    # ---- centre readouts -------------------------------------------------
    d.text((lx + 60, 300), f"{state.speed_knots:.0f}", font=f_big, fill=GREEN)
    d.text((lx + 66, 382), f"GS {state.speed_gs_knots:.0f}", font=f_small, fill=GREEN)
    d.text((rx - 330, 300), f"{state.altitude_ft:.0f}", font=f_big, fill=GREEN)
    d.text((rx - 322, 382), f"VSI {state.vertical_speed_mph:.0f} mph", font=f_small, fill=GREEN)

    # ---- heading ribbon --------------------------------------------------
    base = int(round(state.heading_deg / 5.0) * 5) % 360
    y_ribbon = 470
    d.line([(w // 2 - 260, y_ribbon), (w // 2 + 260, y_ribbon)], fill=GREEN, width=2)
    for k in range(-5, 6):
        x = w // 2 + k * 52
        val = (base + k * 5) % 360
        tick = 20 if k % 2 == 0 else 12
        d.line([(x, y_ribbon - tick // 2), (x, y_ribbon + tick // 2)], fill=GREEN, width=2)
        if k % 2 == 0:
            d.text((x, y_ribbon + 34), f"{val:03d}", font=f_small, fill=GREEN, anchor="ma")
    # Caret pointing at the active heading; the readout hangs below it.
    d.polygon([(w // 2 - 11, y_ribbon - 24), (w // 2 + 11, y_ribbon - 24), (w // 2, y_ribbon - 6)], fill=GREEN)
    d.rectangle([(w // 2 - 46, y_ribbon + 62), (w // 2 + 46, y_ribbon + 106)], outline=GREEN, width=2)
    d.text((w // 2, y_ribbon + 66), f"{state.heading_deg:.0f}", font=f_med, fill=GREEN, anchor="ma")

    # ---- warnings --------------------------------------------------------
    d.text((w // 2 + 170, 415), "Break Off >>", font=f_med, fill=AMBER)
    d.text((w // 2 + 170, 545), "Pull Now >>", font=f_warn, fill=RED)
    d.text((w // 2 + 170, 606), "Minimums >>", font=f_med, fill=RED)

    # ---- hard deck and status -------------------------------------------
    d.text((w // 2 - 420, 118), "HARD DECK", font=f_top, fill=RED)
    d.text((w // 2 - 420, 760), f"{state.profile} {int(62):d}s", font=f_med, fill=GREEN)
    d.text((w // 2 - 40, 760), f"{state.g_load:.1f} G", font=f_med, fill=GREEN)
    d.text((w // 2 + 250, 760), "AAD ARMED" if state.aad_armed else "AAD OFF", font=f_med, fill=AMBER)
    return img


def save_hud(path: str, state: HudState | None = None) -> str:
    img = render_hud(state)
    img.save(path, quality=90)
    return path
