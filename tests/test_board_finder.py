from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import board_finder


class FakePage:
    def __init__(self, failures: int, has_canvas_after_timeout: bool = False) -> None:
        self.failures = failures
        self.has_canvas_after_timeout = has_canvas_after_timeout
        self.goto_calls: list[dict[str, object]] = []
        self.query_count = 0

    def goto(self, url: str, **kwargs: object) -> None:
        self.goto_calls.append({"url": url, **kwargs})
        if len(self.goto_calls) <= self.failures:
            raise TimeoutError("slow navigation")

    def query_selector(self, selector: str) -> object | None:
        self.query_count += 1
        if selector == board_finder.CANVAS_SELECTOR and self.has_canvas_after_timeout:
            return object()
        return None


class BoardFinderNavigationTests(unittest.TestCase):
    def test_open_game_page_retries_after_initial_timeout(self) -> None:
        page = FakePage(failures=1)

        board_finder.open_game_page(page, timeout_ms=1234, attempts=2)

        self.assertEqual(len(page.goto_calls), 2)
        self.assertEqual(page.goto_calls[0]["url"], board_finder.GAME_URL)
        self.assertEqual(page.goto_calls[0]["wait_until"], "commit")
        self.assertEqual(page.goto_calls[0]["timeout"], 1234)

    def test_open_game_page_continues_if_canvas_exists_after_timeout(self) -> None:
        page = FakePage(failures=1, has_canvas_after_timeout=True)

        board_finder.open_game_page(page, timeout_ms=1234, attempts=2)

        self.assertEqual(len(page.goto_calls), 1)
        self.assertEqual(page.query_count, 1)

    def test_open_game_page_raises_actionable_error_after_retries(self) -> None:
        page = FakePage(failures=2)

        with self.assertRaisesRegex(RuntimeError, "Timed out opening the Fruit Box page"):
            board_finder.open_game_page(page, timeout_ms=1234, attempts=2)

        self.assertEqual(len(page.goto_calls), 2)


if __name__ == "__main__":
    unittest.main()
