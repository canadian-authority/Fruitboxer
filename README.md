# Fruitboxer

Fruit Box solver and browser automation helper.

## What it does

- Opens the Fruit Box game in a Chromium browser.
- Clicks Play, screenshots the board, and detects the 17 x 10 apple grid.
- Reads digits with Tesseract OCR.
- Searches for rectangles whose current sum is 10.
- Optionally drags the selected rectangles in the browser.

## Requirements

- Brave, Chromium, or another Playwright-compatible Chromium browser.
- Tesseract OCR, available on PATH or configured with `TESSERACT_CMD`.
- Python dependencies from `requirements.txt`.

The launcher first uses a local `.venv` if one exists. If not, it tries the Windows `py` launcher, then `python` on PATH.

## Run

Preview the board read and planned moves without touching the game:

```bat
run_fruitboxer.bat --no-execute --print-moves
```

Run the full browser solver and execute moves:

```bat
run_fruitboxer.bat --print-moves
```

Solve a saved board JSON without opening the browser:

```bat
run_fruitboxer.bat --board-json examples\sample_board.json --solve-time 2 --seed 1
```

Create a local virtual environment when needed:

```bat
setup_venv.bat
```

## Notes

- Use `--solve-time 10` or higher if you want a better move sequence.
- Use `--seed 123` when you want repeatable solver output.
- Use `--keep-open` to inspect the browser before it closes.
- If your browser is not on PATH, pass `--brave-path path\to\browser.exe` or set `FRUITBOXER_BROWSER`.
