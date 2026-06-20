from __future__ import annotations

"""
Usage:
  python board_finder.py

Dependencies:
  python -m pip install playwright pillow numpy
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image


GAME_URL = "https://en.gamesaien.com/game/fruit_box/"
CANVAS_SELECTOR = "#canvas"
NAVIGATION_ATTEMPTS = 3
BROWSER_ENV_VARS = ("FRUITBOXER_BROWSER", "BRAVE_PATH", "CHROME_PATH")
BROWSER_COMMANDS = (
    "brave",
    "brave.exe",
    "chrome",
    "chrome.exe",
    "chromium",
    "chromium.exe",
    "msedge",
    "msedge.exe",
)
brave_path = os.environ.get("FRUITBOXER_BROWSER") or os.environ.get("BRAVE_PATH")

# The Play button is drawn inside the canvas, so it cannot be clicked by text.
# These ratios are relative to the canvas size and match the game's title screen.
PLAY_BUTTON_X_RATIO = 210 / 720
PLAY_BUTTON_Y_RATIO = 264 / 470

# Fallback crop ratios for the inner board in the game's native 720x470 canvas.
BOARD_FALLBACK = (34 / 720, 39 / 470, 686 / 720, 433 / 470)
LIVE_BOARD_MIN_RED_BLOBS = 80


def smooth(values: np.ndarray, window_size: int) -> np.ndarray:
    """Return a centered moving average without introducing extra dependencies."""
    window_size = max(1, int(window_size))
    if window_size <= 1:
        return values

    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(values, kernel, mode="same")


def close_small_gaps(flags: np.ndarray, max_gap: int) -> np.ndarray:
    closed = flags.copy()
    index = 0

    while index < len(closed):
        if closed[index]:
            index += 1
            continue

        start = index
        while index < len(closed) and not closed[index]:
            index += 1

        gap_len = index - start
        if start > 0 and index < len(closed) and gap_len <= max_gap:
            closed[start:index] = True

    return closed


def longest_true_run(flags: np.ndarray, min_len: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None

    for index, value in enumerate(np.append(flags, False)):
        if value and start is None:
            start = index
            continue

        if not value and start is not None:
            end = index
            if end - start >= min_len:
                if best is None or end - start > best[1] - best[0]:
                    best = (start, end)
            start = None

    return best


def fallback_board_rect(width: int, height: int) -> tuple[int, int, int, int]:
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = BOARD_FALLBACK
    return (
        round(width * x1_ratio),
        round(height * y1_ratio),
        round(width * x2_ratio),
        round(height * y2_ratio),
    )


def detect_board_rect(image: Image.Image) -> tuple[int, int, int, int]:
    """
    Detect the pale green board area inside the Fruit Box canvas.

    The outer frame is saturated green, while the board is mostly pale green and
    white grid cells. Apples create holes in the mask, so the row/column
    projections are smoothed and tiny gaps are closed.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    board_background = (red > 175) & (green > 190) & (blue > 145)

    row_scores = smooth(board_background.mean(axis=1), max(5, height // 45))
    col_scores = smooth(board_background.mean(axis=0), max(5, width // 45))

    row_threshold = max(0.18, float(row_scores.max()) * 0.30)
    col_threshold = max(0.12, float(col_scores.max()) * 0.25)

    row_flags = close_small_gaps(row_scores > row_threshold, max(3, height // 40))
    col_flags = close_small_gaps(col_scores > col_threshold, max(3, width // 40))

    row_span = longest_true_run(row_flags, min_len=height // 2)
    col_span = longest_true_run(col_flags, min_len=width // 2)

    if row_span is None or col_span is None:
        return fallback_board_rect(width, height)

    y1, y2 = row_span
    x1, x2 = col_span

    pad = max(1, round(min(width, height) * 0.003))
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def red_apple_density(image: Image.Image, rect: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = rect
    crop = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    red = crop[:, :, 0].astype(int)
    green = crop[:, :, 1].astype(int)
    blue = crop[:, :, 2].astype(int)
    apple_red = (red > 145) & (green < 135) & (blue < 125) & ((red - green) > 35)
    return float(apple_red.mean())


def red_apple_mask(image: Image.Image, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    crop = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0, 0), dtype=bool)

    red = crop[:, :, 0].astype(int)
    green = crop[:, :, 1].astype(int)
    blue = crop[:, :, 2].astype(int)
    return (red > 145) & (green < 135) & (blue < 125) & ((red - green) > 35)


def count_red_blobs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    min_area = max(25, int(mask.size * 0.00035))
    max_area = max(min_area + 1, int(mask.size * 0.015))
    count = 0

    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue

            stack = [(x, y)]
            seen[y, x] = True
            area = 0

            while stack:
                current_x, current_y = stack.pop()
                area += 1

                for next_x, next_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                ):
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if seen[next_y, next_x] or not mask[next_y, next_x]:
                        continue

                    seen[next_y, next_x] = True
                    stack.append((next_x, next_y))

            if min_area <= area <= max_area:
                count += 1

    return count


def crop_board(canvas_path: Path, board_path: Path) -> tuple[int, int, int, int, float, int]:
    image = Image.open(canvas_path)
    rect = detect_board_rect(image)
    density = red_apple_density(image, rect)
    red_blobs = count_red_blobs(red_apple_mask(image, rect))
    image.crop(rect).save(board_path)
    return (*rect, density, red_blobs)


def candidate_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_candidate{path.suffix}")


def build_board_coordinates(
    page,
    canvas,
    board_rect: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> dict[str, object]:
    canvas_box = canvas.bounding_box()
    if canvas_box is None:
        raise RuntimeError("The game canvas is not visible enough to map coordinates.")

    scroll = page.evaluate("() => ({ x: window.scrollX, y: window.scrollY })")
    x1, y1, x2, y2 = board_rect
    canvas_width, canvas_height = canvas_size
    scale_x = canvas_box["width"] / canvas_width
    scale_y = canvas_box["height"] / canvas_height

    viewport_left = canvas_box["x"] + (x1 * scale_x)
    viewport_top = canvas_box["y"] + (y1 * scale_y)
    viewport_width = (x2 - x1) * scale_x
    viewport_height = (y2 - y1) * scale_y

    return {
        "canvas_pixels": {
            "left": x1,
            "top": y1,
            "right": x2,
            "bottom": y2,
            "width": x2 - x1,
            "height": y2 - y1,
        },
        "viewport_pixels": {
            "left": viewport_left,
            "top": viewport_top,
            "right": viewport_left + viewport_width,
            "bottom": viewport_top + viewport_height,
            "width": viewport_width,
            "height": viewport_height,
        },
        "document_pixels": {
            "left": viewport_left + scroll["x"],
            "top": viewport_top + scroll["y"],
            "right": viewport_left + viewport_width + scroll["x"],
            "bottom": viewport_top + viewport_height + scroll["y"],
            "width": viewport_width,
            "height": viewport_height,
        },
        "canvas_viewport_pixels": canvas_box,
        "page_scroll_pixels": scroll,
    }


def install_hint() -> str:
    return (
        "Missing dependency. Install with:\n"
        "  python -m pip install playwright pillow numpy"
    )


def wait_for_game_canvas(page, timeout_ms: int):
    page.wait_for_selector(CANVAS_SELECTOR, state="visible", timeout=timeout_ms)
    page.wait_for_function(
        """
        () => {
            const canvas = document.querySelector("#canvas");
            const preload = document.querySelector("#_preload_div_");
            const canvasReady = canvas
                && canvas.offsetWidth > 0
                && canvas.offsetHeight > 0
                && getComputedStyle(canvas).display !== "none";
            const preloadDone = !preload || getComputedStyle(preload).display === "none";
            return canvasReady && preloadDone && window.stage && window.exportRoot;
        }
        """,
        timeout=timeout_ms,
    )
    return page.query_selector(CANVAS_SELECTOR)


def open_game_page(page, timeout_ms: int, attempts: int = NAVIGATION_ATTEMPTS) -> None:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    except ImportError:
        PlaywrightTimeoutError = TimeoutError

    timeout_errors = (PlaywrightTimeoutError, TimeoutError)
    attempts = max(1, attempts)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            page.goto(GAME_URL, wait_until="commit", timeout=timeout_ms)
            return
        except timeout_errors as exc:
            last_error = exc
            try:
                if page.query_selector(CANVAS_SELECTOR) is not None:
                    print("Initial navigation timed out, but the game canvas is present; continuing.")
                    return
            except Exception:
                pass

            if attempt >= attempts:
                break

            print(
                "Initial navigation timed out "
                f"after {timeout_ms / 1000:.1f}s; retrying ({attempt + 1}/{attempts})..."
            )
            time.sleep(0.5)

    raise RuntimeError(
        "Timed out opening the Fruit Box page. "
        "Try increasing --timeout or check that https://en.gamesaien.com is reachable."
    ) from last_error


def click_play(page, canvas) -> tuple[float, float]:
    box = canvas.bounding_box()
    if box is None:
        raise RuntimeError("The game canvas is not visible enough to click.")

    play_x = box["x"] + (box["width"] * PLAY_BUTTON_X_RATIO)
    play_y = box["y"] + (box["height"] * PLAY_BUTTON_Y_RATIO)
    page.mouse.click(play_x, play_y)
    return play_x, play_y


def capture_started_board(
    page,
    canvas,
    canvas_path: Path,
    board_path: Path,
    coordinates_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, object], float, int]:
    deadline = time.monotonic() + timeout_seconds
    last_result: tuple[int, int, int, int, float, int] | None = None
    last_play_click = time.monotonic()
    candidate_canvas_path = candidate_path(canvas_path)
    candidate_board_path = candidate_path(board_path)

    while time.monotonic() < deadline:
        canvas.screenshot(path=str(candidate_canvas_path))
        result = crop_board(candidate_canvas_path, candidate_board_path)
        last_result = result
        red_blob_count = result[-1]

        # The title screen has only a few large sample apples. The live board
        # has the dense 17x10 grid, so wait for many separate red apple blobs.
        if red_blob_count >= LIVE_BOARD_MIN_RED_BLOBS:
            candidate_canvas_path.replace(canvas_path)
            candidate_board_path.replace(board_path)
            canvas_image = Image.open(canvas_path)
            board_rect = result[:4]
            coordinates = build_board_coordinates(
                page,
                canvas,
                board_rect,
                canvas_image.size,
            )
            coordinates["detection"] = {
                "apple_red_density": result[-2],
                "red_apple_blobs": red_blob_count,
                "minimum_red_apple_blobs": LIVE_BOARD_MIN_RED_BLOBS,
            }
            coordinates_path.write_text(
                json.dumps(coordinates, indent=2),
                encoding="utf-8",
            )
            return coordinates, result[-2], red_blob_count

        if time.monotonic() - last_play_click >= 1.0:
            play_x, play_y = click_play(page, canvas)
            print(
                "Play not confirmed yet; clicked Play again at "
                f"x={play_x:.1f}, y={play_y:.1f} "
                f"(red apple blobs seen: {red_blob_count})."
            )
            last_play_click = time.monotonic()

        time.sleep(0.25)

    if last_result is None:
        raise RuntimeError("Could not capture the game canvas.")

    canvas_image = Image.open(candidate_canvas_path)
    board_rect = last_result[:4]
    coordinates = build_board_coordinates(page, canvas, board_rect, canvas_image.size)
    coordinates["detection"] = {
        "apple_red_density": last_result[-2],
        "red_apple_blobs": last_result[-1],
        "minimum_red_apple_blobs": LIVE_BOARD_MIN_RED_BLOBS,
    }
    coordinates_path.write_text(json.dumps(coordinates, indent=2), encoding="utf-8")
    raise RuntimeError(
        "Play was not confirmed. The saved screenshot still does not look like "
        f"the live board: found {last_result[-1]} red apple blobs, "
        f"expected at least {LIVE_BOARD_MIN_RED_BLOBS}."
    )


def resolve_browser_executable(configured_path: str | Path | None) -> str | None:
    if configured_path:
        configured = Path(configured_path).expanduser()
        if configured.exists():
            return str(configured)

        configured_command = shutil.which(str(configured_path))
        if configured_command:
            return configured_command

        raise FileNotFoundError(f"Browser executable was not found: {configured_path}")

    for env_var in BROWSER_ENV_VARS:
        configured = os.environ.get(env_var)
        if configured:
            return resolve_browser_executable(configured)

    for command in BROWSER_COMMANDS:
        resolved = shutil.which(command)
        if resolved:
            return resolved

    return None


def launch_browser(playwright, executable_path: str | Path | None, headless: bool):
    browser_type = playwright.chromium
    resolved_executable = resolve_browser_executable(executable_path)

    launch_options = {
        "headless": headless,
        "args": ["--no-first-run"],
    }
    if resolved_executable:
        launch_options["executable_path"] = resolved_executable

    return browser_type.launch(**launch_options)


def run(args: argparse.Namespace) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(install_hint()) from exc

    canvas_path = args.canvas_screenshot.resolve()
    board_path = args.board_screenshot.resolve()
    coordinates_path = args.coordinates.resolve()
    timeout_ms = int(args.timeout * 1000)

    with sync_playwright() as playwright:
        browser = launch_browser(
            playwright,
            executable_path=args.brave_path,
            headless=args.headless,
        )
        page = browser.new_page(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=1,
        )

        print(f"Opening {GAME_URL}")
        open_game_page(page, timeout_ms)
        canvas = wait_for_game_canvas(page, timeout_ms)
        if canvas is None:
            raise RuntimeError("Could not find the game canvas.")

        canvas.scroll_into_view_if_needed(timeout=timeout_ms)
        print("Game loaded. Clicking Play...")
        play_x, play_y = click_play(page, canvas)
        print(f"Pressed Play at viewport coordinates: x={play_x:.1f}, y={play_y:.1f}")

        print("Waiting for the board and taking screenshots...")
        coordinates, density, red_blobs = capture_started_board(
            page,
            canvas,
            canvas_path,
            board_path,
            coordinates_path,
            timeout_seconds=args.timeout,
        )
        canvas_rect = coordinates["canvas_pixels"]
        viewport_rect = coordinates["viewport_pixels"]

        print(f"Saved full canvas screenshot: {canvas_path}")
        print(f"Saved board-area screenshot: {board_path}")
        print(f"Saved board coordinates: {coordinates_path}")
        print(
            "Board rectangle inside canvas: "
            f"left={canvas_rect['left']}, top={canvas_rect['top']}, "
            f"right={canvas_rect['right']}, bottom={canvas_rect['bottom']}"
        )
        print(
            "Board rectangle in viewport: "
            f"left={viewport_rect['left']:.1f}, top={viewport_rect['top']:.1f}, "
            f"right={viewport_rect['right']:.1f}, bottom={viewport_rect['bottom']:.1f}"
        )
        print(f"Detected apple-red density: {density:.3f}")
        print(f"Detected red apple blobs: {red_blobs}")

        if args.keep_open:
            input("Browser is still open. Press Enter to close it...")

        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open Fruit Box, click Play, detect the board, and screenshot it."
    )
    parser.add_argument(
        "--canvas-screenshot",
        type=Path,
        default=Path("fruit_box_canvas.png"),
        help="Where to save the full game canvas screenshot.",
    )
    parser.add_argument(
        "--board-screenshot",
        type=Path,
        default=Path("fruit_box_board.png"),
        help="Where to save the detected board-area screenshot.",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=Path("fruit_box_board_coordinates.json"),
        help="Where to save the detected board coordinates.",
    )
    parser.add_argument(
        "--brave-path",
        default=brave_path,
        help=(
            "Path to a Chromium browser executable. If omitted, uses "
            "FRUITBOXER_BROWSER, BRAVE_PATH, a browser on PATH, or Playwright Chromium."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without showing the browser window.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave the browser open until you press Enter.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the page and board.",
    )
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
