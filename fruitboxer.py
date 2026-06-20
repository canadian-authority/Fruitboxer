from __future__ import annotations

import argparse
import io
import json
import os
import time
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import board_finder

# =======================================================================
# DEPENDENCIES REQUIRED:
# pip install playwright pillow numpy
# 
# Install Tesseract OCR separately. The default Windows path is checked below.
# =======================================================================

DIGIT_WHITELIST = "123456789"
TEMPLATE_CLUSTER_MAX_MSE = 0.01
TESSERACT_ENV_VARS = ("TESSERACT_CMD", "TESSERACT_PATH")

# --- 0. INTERPOLATION HELPER ---
def calculate_apple_centers(TL, TR, BR, BL, cols=17, rows=10):
    """
    Uses the 4 corner coordinates to calculate the exact pixel center 
    for every single apple on the board, handling any non-square stretching.
    """
    centers = {}
    for r in range(rows):
        for c in range(cols):
            # 1. Interpolate horizontally along the top and bottom edges
            u = c / (cols - 1)
            top_x = TL[0] + (TR[0] - TL[0]) * u
            top_y = TL[1] + (TR[1] - TL[1]) * u
            
            bot_x = BL[0] + (BR[0] - BL[0]) * u
            bot_y = BL[1] + (BR[1] - BL[1]) * u

            # 2. Interpolate vertically between those two points
            v = r / (rows - 1)
            x = top_x + (bot_x - top_x) * v
            y = top_y + (bot_y - top_y) * v

            centers[(r, c)] = (int(x), int(y))
    return centers

# --- 1. DETECTOR FUNCTION ---
@dataclass
class Glyph:
    box: tuple[int, int, int, int]
    area: int
    cx: float
    cy: float
    normalized: np.ndarray
    digit: str | None = None
    score: float | None = None


def find_tesseract() -> str:
    for env_var in TESSERACT_ENV_VARS:
        configured = os.environ.get(env_var)
        if not configured:
            continue

        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path)

        configured_command = shutil.which(configured)
        if configured_command:
            return configured_command

        raise RuntimeError(
            f"{env_var} points to {configured!r}, but that executable was not found."
        )

    tesseract = shutil.which("tesseract")
    if tesseract:
        return tesseract
    raise RuntimeError(
        "Could not find tesseract. Install it, add it to PATH, or set TESSERACT_CMD."
    )


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def foreground_mask(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"))
    threshold = otsu_threshold(gray)
    mask = gray <= threshold

    # If an image is inverted, foreground should still be the minority color.
    if mask.mean() > 0.5:
        mask = ~mask

    return mask


def connected_components(mask: np.ndarray, min_area: int = 20) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    neighbors = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    for y in range(height):
        for x in range(width):
            if seen[y, x] or not mask[y, x]:
                continue

            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []

            while stack:
                current_x, current_y = stack.pop()
                xs.append(current_x)
                ys.append(current_y)

                for dx, dy in neighbors:
                    next_x = current_x + dx
                    next_y = current_y + dy
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if seen[next_y, next_x] or not mask[next_y, next_x]:
                        continue

                    seen[next_y, next_x] = True
                    stack.append((next_x, next_y))

            area = len(xs)
            if area >= min_area:
                components.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, area))

    return components


def normalize_glyph(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    size: int = 32,
    pad: int = 3,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    ys, xs = np.where(crop)
    if len(xs) == 0:
        return np.zeros((size, size), dtype=np.float32)

    crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    crop_height, crop_width = crop.shape
    scale = (size - (pad * 2)) / max(crop_height, crop_width)
    new_width = max(1, round(crop_width * scale))
    new_height = max(1, round(crop_height * scale))

    resized = Image.fromarray((crop * 255).astype("uint8")).resize(
        (new_width, new_height),
        Image.Resampling.NEAREST,
    )

    canvas = np.zeros((size, size), dtype=np.float32)
    offset_x = (size - new_width) // 2
    offset_y = (size - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = (
        np.array(resized) > 0
    )
    return canvas


def extract_glyphs(image_path: str | Path) -> tuple[Image.Image, list[Glyph]]:
    image = Image.open(image_path)
    mask = foreground_mask(image)
    glyphs: list[Glyph] = []
    min_area = max(4, int(mask.size * 0.00002))

    for x1, y1, x2, y2, area in connected_components(mask, min_area=min_area):
        glyphs.append(
            Glyph(
                box=(x1, y1, x2, y2),
                area=area,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                normalized=normalize_glyph(mask, (x1, y1, x2, y2)),
            )
        )

    if not glyphs:
        raise RuntimeError("No digit components were found in the image.")

    return image, glyphs


def tesseract_boxes(image_path: str | Path, image_height: int) -> list[tuple[str, int, int, int, int]]:
    result = subprocess.run(
        [
            find_tesseract(),
            str(image_path),
            "stdout",
            "--psm",
            "6",
            "--oem",
            "3",
            "-c",
            f"tessedit_char_whitelist={DIGIT_WHITELIST}",
            "makebox",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    boxes: list[tuple[str, int, int, int, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] not in DIGIT_WHITELIST:
            continue

        digit = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:5])
        boxes.append((digit, x1, image_height - y2, x2, image_height - y1))

    return boxes


def build_templates(
    glyphs: list[Glyph],
    boxes: list[tuple[str, int, int, int, int]],
    margin: int = 2,
) -> dict[str, list[np.ndarray]]:
    clusters: list[dict[str, object]] = []

    for digit, x1, y1, x2, y2 in boxes:
        matches = [
            glyph
            for glyph in glyphs
            if x1 - margin <= glyph.cx <= x2 + margin
            and y1 - margin <= glyph.cy <= y2 + margin
        ]

        # Wide OCR boxes can span two real glyphs, so use only one-to-one labels.
        if len(matches) == 1:
            sample = matches[0].normalized
            for cluster in clusters:
                representative = cluster["representative"]
                score = float(np.mean((sample - representative) ** 2))
                if score <= TEMPLATE_CLUSTER_MAX_MSE:
                    samples = cluster["samples"]
                    votes = cluster["votes"]
                    samples.append(sample)
                    votes[digit] += 1
                    cluster["representative"] = np.mean(samples, axis=0)
                    break
            else:
                clusters.append(
                    {
                        "representative": sample.copy(),
                        "samples": [sample],
                        "votes": Counter({digit: 1}),
                    }
                )

    templates: dict[str, list[np.ndarray]] = defaultdict(list)
    for cluster in clusters:
        votes = cluster["votes"]
        samples = cluster["samples"]
        best_digit = min(
            votes,
            key=lambda digit: (-votes[digit], DIGIT_WHITELIST.index(digit)),
        )
        templates[best_digit].append(np.mean(samples, axis=0))

    if not templates:
        raise RuntimeError("Tesseract did not produce any usable digit templates.")

    return templates


def classify_glyphs(glyphs: list[Glyph], templates: dict[str, list[np.ndarray]]) -> None:
    for glyph in glyphs:
        best_digit: str | None = None
        best_score = float("inf")

        for digit, samples in templates.items():
            for sample in samples:
                score = float(np.mean((glyph.normalized - sample) ** 2))
                if score < best_score:
                    best_score = score
                    best_digit = digit

        glyph.digit = best_digit
        glyph.score = best_score


def group_rows(glyphs: list[Glyph]) -> list[list[Glyph]]:
    median_height = float(np.median([glyph.box[3] - glyph.box[1] for glyph in glyphs]))
    max_same_row_gap = median_height * 1.5
    rows: list[list[Glyph]] = []

    for glyph in sorted(glyphs, key=lambda item: item.cy):
        if not rows:
            rows.append([glyph])
            continue

        row_center = float(np.mean([item.cy for item in rows[-1]]))
        if abs(glyph.cy - row_center) > max_same_row_gap:
            rows.append([glyph])
        else:
            rows[-1].append(glyph)

    for row in rows:
        row.sort(key=lambda item: item.cx)

    return rows


def estimate_column_centers(glyphs: list[Glyph], num_columns: int) -> list[float]:
    widths = [glyph.box[2] - glyph.box[0] for glyph in glyphs]
    max_same_column_gap = float(np.median(widths)) * 1.5
    clusters: list[list[float]] = []

    for glyph in sorted(glyphs, key=lambda item: item.cx):
        if not clusters or abs(glyph.cx - float(np.mean(clusters[-1]))) > max_same_column_gap:
            clusters.append([glyph.cx])
        else:
            clusters[-1].append(glyph.cx)

    if len(clusters) == num_columns:
        return [float(np.mean(cluster)) for cluster in clusters]

    centers = sorted(glyph.cx for glyph in glyphs)
    if len(centers) < num_columns:
        raise RuntimeError(f"Only found {len(centers)} glyphs; expected at least {num_columns}.")

    return [float(np.mean(chunk)) for chunk in np.array_split(centers, num_columns)]


def extract_matrix(image_path: str | Path, num_columns: int = 17) -> list[list[int | str]]:
    image, glyphs = extract_glyphs(image_path)
    boxes = tesseract_boxes(image_path, image.height)
    templates = build_templates(glyphs, boxes)
    classify_glyphs(glyphs, templates)

    column_centers = estimate_column_centers(glyphs, num_columns)
    matrix: list[list[int | str]] = []

    for row in group_rows(glyphs):
        formatted_row: list[int | str] = ["?"] * num_columns
        for glyph in row:
            distances = [abs(glyph.cx - center) for center in column_centers]
            column = int(np.argmin(distances))
            formatted_row[column] = int(glyph.digit) if glyph.digit else "?"
        matrix.append(formatted_row)

    return matrix


def recognize_board_rows(image_path: str | Path, cols: int = 17, rows: int = 10):
    try:
        matrix = extract_matrix(image_path, cols)
    except Exception as exc:
        print(f"OCR failed: {exc}")
        return None

    if len(matrix) != rows:
        print(f"OCR failed: expected {rows} rows, found {len(matrix)}.")
        return None

    board = []
    for row_index, row in enumerate(matrix):
        if len(row) != cols:
            print(f"OCR failed: row {row_index + 1} has {len(row)} columns, expected {cols}.")
            return None

        parsed_row = []
        for col_index, value in enumerate(row):
            if isinstance(value, int) and 1 <= value <= 9:
                parsed_row.append(value)
                continue

            print(f"OCR failed: unreadable digit at row {row_index + 1}, column {col_index + 1}.")
            return None

        board.append(parsed_row)

    return board


def build_digit_vision_image(
    image: Image.Image,
    blue_threshold: int = 239,
    cols: int = 17,
    rows: int = 10,
) -> Image.Image:
    """
    Convert the Fruit Box board crop into a high-contrast OCR image.

    The board_finder crop includes faint white grid lines, so prefer extracting
    white glyph components close to each red apple center. The blue-channel
    fallback keeps the legacy screenshot path working if apple detection fails.
    """
    try:
        return build_digit_vision_from_apples(image, cols, rows)
    except RuntimeError:
        pass

    rgb = np.asarray(image.convert("RGB"))
    digit_mask = rgb[:, :, 2] > blue_threshold
    vision = np.where(digit_mask, 0, 255).astype("uint8")
    return Image.fromarray(vision)


def save_digit_vision_image(
    image_or_path: Image.Image | str | Path,
    output_path: str | Path,
    blue_threshold: int = 239,
    cols: int = 17,
    rows: int = 10,
) -> Path:
    image = (
        Image.open(image_or_path)
        if isinstance(image_or_path, (str, Path))
        else image_or_path
    )
    vision_path = Path(output_path)
    build_digit_vision_image(image, blue_threshold, cols, rows).save(vision_path)
    return vision_path


def red_apple_blob_centers_from_image(image: Image.Image) -> list[tuple[float, float]]:
    mask = board_finder.red_apple_mask(image, (0, 0, *image.size))
    min_area = max(25, int(mask.size * 0.00035))
    max_area = max(min_area + 1, int(mask.size * 0.015))
    centers: list[tuple[float, float]] = []

    for x1, y1, x2, y2, area in connected_components(mask, min_area=min_area):
        if area > max_area:
            continue

        crop = mask[y1:y2, x1:x2]
        ys, xs = np.where(crop)
        if len(xs) == 0:
            continue

        centers.append((float(x1 + np.mean(xs)), float(y1 + np.mean(ys))))

    return centers


def red_apple_blob_centers(board_image_path: str | Path) -> list[tuple[float, float]]:
    return red_apple_blob_centers_from_image(Image.open(board_image_path))


def apple_center_rows(
    centers: list[tuple[float, float]],
    cols: int = 17,
    rows: int = 10,
) -> list[list[tuple[float, float]]]:
    expected = cols * rows
    if len(centers) != expected:
        raise RuntimeError(f"Detected {len(centers)} apple centers; expected {expected}.")

    by_y = sorted(centers, key=lambda item: item[1])
    grid_rows: list[list[tuple[float, float]]] = []
    for row_index in range(rows):
        row = by_y[row_index * cols : (row_index + 1) * cols]
        if len(row) != cols:
            raise RuntimeError(
                f"Detected row {row_index + 1} has {len(row)} apples; expected {cols}."
            )
        grid_rows.append(sorted(row, key=lambda item: item[0]))

    return grid_rows


def estimate_grid_steps(grid_rows: list[list[tuple[float, float]]]) -> tuple[float, float]:
    horizontal_steps = []
    vertical_steps = []

    for row in grid_rows:
        for left, right in zip(row, row[1:]):
            horizontal_steps.append(abs(right[0] - left[0]))

    for top, bottom in zip(grid_rows, grid_rows[1:]):
        top_y = float(np.mean([center[1] for center in top]))
        bottom_y = float(np.mean([center[1] for center in bottom]))
        vertical_steps.append(abs(bottom_y - top_y))

    return (
        float(np.median(horizontal_steps)) if horizontal_steps else 33.0,
        float(np.median(vertical_steps)) if vertical_steps else 33.0,
    )


def build_digit_vision_from_apples(
    image: Image.Image,
    cols: int = 17,
    rows: int = 10,
) -> Image.Image:
    grid_rows = apple_center_rows(red_apple_blob_centers_from_image(image), cols, rows)
    step_x, step_y = estimate_grid_steps(grid_rows)
    radius_x = max(6, int(step_x * 0.47))
    radius_y = max(6, int(step_y * 0.47))
    max_center_dx = max(3.0, radius_x * 0.45)
    max_center_dy = max(4.0, radius_y * 0.55)

    rgb = np.asarray(image.convert("RGB"))
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    white_mask = (red > 230) & (green > 230) & (blue > 200)
    vision = np.full(white_mask.shape, 255, dtype="uint8")

    for row in grid_rows:
        for center_x, center_y in row:
            x1 = max(0, int(center_x - radius_x))
            x2 = min(image.width, int(center_x + radius_x + 1))
            y1 = max(0, int(center_y - radius_y))
            y2 = min(image.height, int(center_y + radius_y + 1))
            crop = white_mask[y1:y2, x1:x2]

            for bx1, by1, bx2, by2, area in connected_components(crop, min_area=2):
                component_center_x = x1 + ((bx1 + bx2) / 2)
                component_center_y = y1 + ((by1 + by2) / 2)
                if abs(component_center_x - center_x) > max_center_dx:
                    continue
                if abs(component_center_y - center_y) > max_center_dy:
                    continue
                if area < 4:
                    continue

                target = vision[y1 + by1 : y1 + by2, x1 + bx1 : x1 + bx2]
                target[crop[by1:by2, bx1:bx2]] = 0

    return Image.fromarray(vision)


def map_detected_apple_centers(
    board_image_path: str | Path,
    coordinates: dict[str, object],
    cols: int = 17,
    rows: int = 10,
) -> dict[tuple[int, int], tuple[float, float]]:
    image = Image.open(board_image_path)
    grid_rows = apple_center_rows(red_apple_blob_centers_from_image(image), cols, rows)

    viewport = coordinates["viewport_pixels"]
    left = float(viewport["left"])
    top = float(viewport["top"])
    scale_x = float(viewport["width"]) / image.width
    scale_y = float(viewport["height"]) / image.height

    mapped: dict[tuple[int, int], tuple[float, float]] = {}
    for row_index, row in enumerate(grid_rows):
        for col_index, (local_x, local_y) in enumerate(row):
            mapped[(row_index, col_index)] = (
                left + (local_x * scale_x),
                top + (local_y * scale_y),
            )

    return mapped


def detect_board(centers, TL, TR, BR, BL, cols=17, rows=10):
    # Calculate a bounding box that covers all 4 corners + 30 pixels of padding
    import pyautogui

    padding = 30
    left = min(TL[0], BL[0]) - padding
    right = max(TR[0], BR[0]) + padding
    top = min(TL[1], TR[1]) - padding
    bottom = max(BL[1], BR[1]) + padding
    
    width = right - left
    height = bottom - top
    
    # Take ONE screenshot of the calculated area
    img = pyautogui.screenshot(region=(left, top, width, height))
    img.save("debug_raw.png")
    vision_path = Path("debug_vision.png")
    save_digit_vision_image(img, vision_path, cols=cols, rows=rows)
    
    return recognize_board_rows(vision_path, cols, rows)

# --- 2. SOLVER FUNCTION ---
Move = tuple[int, int, int, int]
ScoredMove = tuple[int, int, int, int, int, int]


def flatten_board(board: list[list[int]]) -> tuple[list[int], int, int]:
    if not board or not board[0]:
        raise ValueError("Board must contain at least one row and one column.")

    rows = len(board)
    cols = len(board[0])
    grid: list[int] = []

    for row_index, row in enumerate(board):
        if len(row) != cols:
            raise ValueError(
                f"Board row {row_index + 1} has {len(row)} columns; expected {cols}."
            )

        for col_index, value in enumerate(row):
            number = int(value)
            if number < 0 or number > 9:
                raise ValueError(
                    "Board values must be digits from 0 to 9; "
                    f"got {value!r} at row {row_index + 1}, column {col_index + 1}."
                )
            grid.append(number)

    return grid, rows, cols


def build_prefix_tables(
    grid: list[int],
    rows: int,
    cols: int,
) -> tuple[list[list[int]], list[list[int]]]:
    sums = [[0] * (cols + 1) for _ in range(rows + 1)]
    counts = [[0] * (cols + 1) for _ in range(rows + 1)]

    for r in range(rows):
        for c in range(cols):
            value = grid[r * cols + c]
            sums[r + 1][c + 1] = (
                value + sums[r][c + 1] + sums[r + 1][c] - sums[r][c]
            )
            counts[r + 1][c + 1] = (
                (1 if value > 0 else 0)
                + counts[r][c + 1]
                + counts[r + 1][c]
                - counts[r][c]
            )

    return sums, counts


def rect_total(prefix: list[list[int]], r1: int, r2: int, c1: int, c2: int) -> int:
    return prefix[r2][c2] - prefix[r1][c2] - prefix[r2][c1] + prefix[r1][c1]


def find_valid_moves(grid: list[int], rows: int, cols: int) -> list[ScoredMove]:
    sums, counts = build_prefix_tables(grid, rows, cols)
    moves: list[ScoredMove] = []

    for r1 in range(rows):
        for r2 in range(r1 + 1, rows + 1):
            for c1 in range(cols):
                for c2 in range(c1 + 1, cols + 1):
                    total = rect_total(sums, r1, r2, c1, c2)
                    if total == 10:
                        apples = rect_total(counts, r1, r2, c1, c2)
                        area = (r2 - r1) * (c2 - c1)
                        moves.append((r1, r2, c1, c2, apples, area))
                    elif total > 10:
                        break

    return moves


def apply_move(grid: list[int], cols: int, move: Move | ScoredMove) -> int:
    r1, r2, c1, c2 = move[:4]
    removed = 0

    for r in range(r1, r2):
        for c in range(c1, c2):
            index = r * cols + c
            if grid[index] > 0:
                removed += 1
                grid[index] = 0

    return removed


def choose_move(
    candidates: list[ScoredMove],
    rng: random.Random,
    exploratory: bool,
) -> ScoredMove:
    candidates.sort(key=lambda move: (move[4], move[5], -move[0], -move[2]), reverse=True)
    if not exploratory or len(candidates) == 1:
        return candidates[0]

    pool = candidates[: min(len(candidates), rng.randint(4, 18))]
    total_weight = sum(max(1, move[4]) ** 3 for move in pool)
    pick = rng.uniform(0, total_weight)
    running = 0.0

    for move in pool:
        running += max(1, move[4]) ** 3
        if running >= pick:
            return move

    return pool[-1]


def solve_one_pass(
    initial_grid: list[int],
    rows: int,
    cols: int,
    rng: random.Random,
    exploratory: bool,
) -> tuple[int, list[Move]]:
    grid = initial_grid.copy()
    current_score = 0
    moves: list[Move] = []

    while True:
        candidates = find_valid_moves(grid, rows, cols)
        if not candidates:
            return current_score, moves

        move = choose_move(candidates, rng, exploratory)
        removed = apply_move(grid, cols, move)
        if removed == 0:
            return current_score, moves

        moves.append(move[:4])
        current_score += removed


def solve_fruit_box(
    board: list[list[int]],
    time_limit: float = 20.0,
    seed: int | None = None,
) -> tuple[int, list[Move]]:
    initial_grid, rows, cols = flatten_board(board)
    total_apples = sum(1 for value in initial_grid if value > 0)
    if total_apples == 0:
        return 0, []

    best_score = 0
    best_moves: list[Move] = []
    rng = random.Random(seed)
    deadline = time.monotonic() + max(0.0, time_limit)
    run_index = 0

    while True:
        score, moves = solve_one_pass(
            initial_grid,
            rows,
            cols,
            rng,
            exploratory=run_index > 0,
        )

        if score > best_score:
            best_score = score
            best_moves = moves
            if best_score == total_apples:
                break

        if not moves:
            break

        run_index += 1
        if run_index > 0 and time.monotonic() >= deadline:
            break

    return best_score, best_moves


def load_board_json(path: Path) -> list[list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("board", data.get("rows"))

    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise ValueError("Board JSON must be a 2D array or an object with a board array.")

    board = [[int(value) for value in row] for row in data]
    flatten_board(board)
    return board


def print_move_sequence(moves: list[Move]) -> None:
    if not moves:
        print("No moves found.")
        return

    print("Moves, using 1-based inclusive rows and columns:")
    for index, (r1, r2, c1, c2) in enumerate(moves, start=1):
        print(f"{index:02d}. rows {r1 + 1}-{r2}, cols {c1 + 1}-{c2}")


def solve_board_file(args: argparse.Namespace) -> None:
    board = load_board_json(args.board_json.resolve())
    print("Loaded Board:")
    for row in board:
        print(row)

    best_pts, best_seq = solve_fruit_box(
        board,
        time_limit=args.solve_time,
        seed=args.seed,
    )
    print(f"\nFound a sequence worth {best_pts} points.")
    print_move_sequence(best_seq)

# --- 3. AUTOMATION FUNCTION ---
def estimate_drag_padding(centers):
    """
    Pick drag padding from the detected grid spacing instead of fixed pixels.
    This keeps the drag outside the edge apples without reaching neighboring
    apple centers when the browser zoom changes.
    """
    horizontal_steps = []
    vertical_steps = []
    for (r, c), (x, y) in centers.items():
        if (r, c + 1) in centers:
            next_x, _ = centers[(r, c + 1)]
            horizontal_steps.append(abs(next_x - x))
        if (r + 1, c) in centers:
            _, next_y = centers[(r + 1, c)]
            vertical_steps.append(abs(next_y - y))

    horizontal_step = float(np.median(horizontal_steps)) if horizontal_steps else 86.0
    vertical_step = float(np.median(vertical_steps)) if vertical_steps else 86.0

    padding_x = int(max(4, min(horizontal_step * 0.78, horizontal_step - 8)))
    padding_y = int(max(4, min(vertical_step * 0.78, vertical_step - 8)))
    return padding_x, padding_y


def drag_to_select(start_x, start_y, end_x, end_y):
    import pyautogui

    pyautogui.moveTo(start_x, start_y, duration=0.05)
    time.sleep(0.05)
    pyautogui.mouseDown(button='left')
    time.sleep(0.05)
    pyautogui.moveTo(end_x, end_y, duration=0.55)
    time.sleep(0.12)
    pyautogui.mouseUp(button='left')


def execute_moves_in_browser(moves, centers):
    import pyautogui

    pyautogui.FAILSAFE = True 
    print("Executing moves...")
    padding_x, padding_y = estimate_drag_padding(centers)
    
    for r1, r2, c1, c2 in moves:
        # Drag from grid cell bounds, not OCR digit centers. Tesseract's
        # character boxes can sit a few pixels inside an apple and miss an edge.
        start_x, start_y = centers[(r1, c1)]
        end_x, end_y = centers[(r2 - 1, c2 - 1)]
        
        start_x -= padding_x
        start_y -= padding_y
        
        end_x += padding_x
        end_y += padding_y
        
        drag_to_select(start_x, start_y, end_x, end_y)
        time.sleep(0.1) 
        
    print("Finished execution!")


def drag_to_select_in_page(page, start_x, start_y, end_x, end_y):
    page.mouse.move(start_x, start_y)
    time.sleep(0.05)
    page.mouse.down(button="left")
    time.sleep(0.05)
    page.mouse.move(end_x, end_y, steps=16)
    time.sleep(0.12)
    page.mouse.up(button="left")


def execute_moves_in_page(page, moves, centers):
    print("Executing moves...")
    padding_x, padding_y = estimate_drag_padding(centers)
    
    for r1, r2, c1, c2 in moves:
        start_x, start_y = centers[(r1, c1)]
        end_x, end_y = centers[(r2 - 1, c2 - 1)]
        
        start_x -= padding_x
        start_y -= padding_y
        
        end_x += padding_x
        end_y += padding_y
        
        drag_to_select_in_page(page, start_x, start_y, end_x, end_y)
        time.sleep(0.1) 
        
    print("Finished execution!")

# --- 4. CONFIGURATION & RUN ---
def capture_live_board(playwright, args: argparse.Namespace):
    canvas_path = args.canvas_screenshot.resolve()
    board_path = args.board_screenshot.resolve()
    coordinates_path = args.coordinates.resolve()
    timeout_ms = int(args.timeout * 1000)

    browser = board_finder.launch_browser(
        playwright,
        executable_path=args.brave_path,
        headless=args.headless,
    )
    try:
        page = browser.new_page(
            viewport={"width": args.viewport_width, "height": args.viewport_height},
            device_scale_factor=1,
        )

        print(f"Opening {board_finder.GAME_URL}")
        board_finder.open_game_page(page, timeout_ms)
        canvas = board_finder.wait_for_game_canvas(page, timeout_ms)
        if canvas is None:
            raise RuntimeError("Could not find the game canvas.")

        canvas.scroll_into_view_if_needed(timeout=timeout_ms)
        print("Game loaded. Clicking Play...")
        play_x, play_y = board_finder.click_play(page, canvas)
        print(f"Pressed Play at viewport coordinates: x={play_x:.1f}, y={play_y:.1f}")

        print("Waiting for the board and taking screenshots...")
        coordinates, density, red_blobs = board_finder.capture_started_board(
            page,
            canvas,
            canvas_path,
            board_path,
            coordinates_path,
            timeout_seconds=args.timeout,
        )

        print(f"Saved full canvas screenshot: {canvas_path}")
        print(f"Saved board-area screenshot: {board_path}")
        print(f"Saved board coordinates: {coordinates_path}")
        print(f"Detected apple-red density: {density:.3f}")
        print(f"Detected red apple blobs: {red_blobs}")
        return browser, page, coordinates, board_path
    except Exception:
        browser.close()
        raise


def read_live_board(board_path: Path, vision_path: Path, cols: int, rows: int):
    save_digit_vision_image(board_path, vision_path, cols=cols, rows=rows)

    for attempt in range(1, 4):
        live_board = recognize_board_rows(vision_path, cols, rows)
        if live_board is not None:
            return live_board

        if attempt < 3:
            print(f"Retrying OCR after failed attempt {attempt}...")
            time.sleep(0.3)

    return None


def save_current_board_screenshot(page, coordinates: dict[str, object], output_path: Path) -> Path:
    canvas = page.query_selector(board_finder.CANVAS_SELECTOR)
    if canvas is None:
        raise RuntimeError("Could not find the game canvas for the final screenshot.")

    rect = coordinates["canvas_pixels"]
    image = Image.open(io.BytesIO(canvas.screenshot()))
    left = max(0, int(round(float(rect["left"]))))
    top = max(0, int(round(float(rect["top"]))))
    right = min(image.width, int(round(float(rect["right"]))))
    bottom = min(image.height, int(round(float(rect["bottom"]))))
    if right <= left or bottom <= top:
        raise RuntimeError("The saved board coordinates are not valid for the final screenshot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).save(output_path)
    return output_path


def run(args: argparse.Namespace) -> None:
    if args.board_json is not None:
        solve_board_file(args)
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(board_finder.install_hint()) from exc

    with sync_playwright() as playwright:
        browser = None
        page = None
        coordinates = None
        try:
            print("\n--- Phase 1: Finding Board ---")
            browser, page, coordinates, board_path = capture_live_board(playwright, args)
            apple_centers_map = map_detected_apple_centers(
                board_path,
                coordinates,
                cols=args.cols,
                rows=args.rows,
            )

            print("\n--- Phase 2: Reading Board ---")
            live_board = read_live_board(
                board_path,
                args.vision_screenshot.resolve(),
                args.cols,
                args.rows,
            )
            if live_board is None:
                print("\nERROR: Could not read a complete board. No moves will be executed.")
                if args.keep_open:
                    input("Browser is still open. Press Enter to close it...")
                return

            print("Detected Board:")
            for row in live_board:
                print(row)

            print("\n--- Phase 3: Running Solver ---")
            best_pts, best_seq = solve_fruit_box(
                live_board,
                time_limit=args.solve_time,
                seed=args.seed,
            )
            print(f"Found a sequence worth {best_pts} points.")

            if args.no_execute:
                print("\nMove execution skipped.")
            if args.no_execute or args.print_moves:
                print_move_sequence(best_seq)
            if not args.no_execute:
                print("\n--- Phase 4: Executing Moves ---")
                execute_moves_in_page(page, best_seq, apple_centers_map)

            if args.keep_open:
                input("Browser is still open. Press Enter to close it...")
        finally:
            if browser is not None:
                try:
                    if page is not None and coordinates is not None and not page.is_closed():
                        final_board_path = save_current_board_screenshot(
                            page,
                            coordinates,
                            args.final_board_screenshot.resolve(),
                        )
                        print(f"Saved final board screenshot: {final_board_path}")
                except Exception as exc:
                    print(f"WARNING: Could not save final board screenshot: {exc}")
                finally:
                    browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open Fruit Box, detect the board, solve it, and execute moves."
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
        "--final-board-screenshot",
        type=Path,
        default=Path("fruit_box_board_final.png"),
        help="Where to save the board screenshot captured immediately before closing.",
    )
    parser.add_argument(
        "--vision-screenshot",
        type=Path,
        default=Path("debug_vision.png"),
        help="Where to save the high-contrast OCR image.",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=Path("fruit_box_board_coordinates.json"),
        help="Where to save the detected board coordinates.",
    )
    parser.add_argument(
        "--brave-path",
        default=board_finder.brave_path,
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
        "--no-execute",
        action="store_true",
        help="Detect and solve the board without dragging moves.",
    )
    parser.add_argument(
        "--board-json",
        type=Path,
        help="Solve a board saved as a JSON 2D array instead of opening the browser.",
    )
    parser.add_argument(
        "--print-moves",
        action="store_true",
        help="Print the selected move sequence before execution.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the page and board.",
    )
    parser.add_argument(
        "--solve-time",
        type=float,
        default=5.0,
        help="Seconds to spend searching for a high-scoring move sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible solver restarts.",
    )
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--cols", type=int, default=17)
    parser.add_argument("--rows", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
