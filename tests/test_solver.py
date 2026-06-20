from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitboxer import apply_move, flatten_board, solve_fruit_box


class FruitBoxSolverTests(unittest.TestCase):
    def test_solves_simple_board(self) -> None:
        board = [
            [5, 5],
            [4, 6],
        ]

        score, moves = solve_fruit_box(board, time_limit=0, seed=1)

        self.assertEqual(score, 4)
        self.assertEqual(len(moves), 2)

    def test_returned_moves_are_valid_when_applied_in_order(self) -> None:
        board = [
            [5, 5, 2, 8],
            [3, 2, 9, 1],
            [7, 1, 1, 5],
        ]
        grid, _, cols = flatten_board(board)

        score, moves = solve_fruit_box(board, time_limit=0.1, seed=7)
        removed = 0

        for r1, r2, c1, c2 in moves:
            total = sum(
                grid[r * cols + c]
                for r in range(r1, r2)
                for c in range(c1, c2)
            )
            self.assertEqual(total, 10)
            removed += apply_move(grid, cols, (r1, r2, c1, c2))

        self.assertEqual(score, removed)


if __name__ == "__main__":
    unittest.main()
