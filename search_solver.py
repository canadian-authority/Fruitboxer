from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

Move = tuple[int, int, int, int]
ScoredMove = tuple[int, int, int, int, int, int]
BoardState = tuple[int, ...]


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    score: int
    moves: list[Move]
    elapsed: float


def flatten_board(board: list[list[int]]) -> tuple[BoardState, int, int]:
    if not board or not board[0]:
        raise ValueError("Board must contain at least one row and one column.")

    rows = len(board)
    cols = len(board[0])
    values: list[int] = []

    for row_index, row in enumerate(board):
        if len(row) != cols:
            raise ValueError(
                f"Board row {row_index + 1} has {len(row)} columns; expected {cols}."
            )

        for col_index, raw_value in enumerate(row):
            value = int(raw_value)
            if value < 0 or value > 9:
                raise ValueError(
                    "Board values must be digits from 0 to 9; "
                    f"got {raw_value!r} at row {row_index + 1}, column {col_index + 1}."
                )
            values.append(value)

    return tuple(values), rows, cols


def load_board_json(path: Path) -> list[list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("board", data.get("rows"))

    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise ValueError("Board JSON must be a 2D array or an object with a board array.")

    board = [[int(value) for value in row] for row in data]
    flatten_board(board)
    return board


def build_prefix_tables(
    state: BoardState,
    rows: int,
    cols: int,
) -> tuple[list[list[int]], list[list[int]]]:
    sums = [[0] * (cols + 1) for _ in range(rows + 1)]
    counts = [[0] * (cols + 1) for _ in range(rows + 1)]

    for row in range(rows):
        for col in range(cols):
            value = state[row * cols + col]
            sums[row + 1][col + 1] = (
                value + sums[row][col + 1] + sums[row + 1][col] - sums[row][col]
            )
            counts[row + 1][col + 1] = (
                (1 if value > 0 else 0)
                + counts[row][col + 1]
                + counts[row + 1][col]
                - counts[row][col]
            )

    return sums, counts


def rect_total(prefix: list[list[int]], r1: int, r2: int, c1: int, c2: int) -> int:
    return prefix[r2][c2] - prefix[r1][c2] - prefix[r2][c1] + prefix[r1][c1]


def find_valid_moves(state: BoardState, rows: int, cols: int) -> list[ScoredMove]:
    sums, counts = build_prefix_tables(state, rows, cols)
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


def apply_move(state: BoardState, cols: int, move: Move | ScoredMove) -> tuple[BoardState, int]:
    r1, r2, c1, c2 = move[:4]
    values = list(state)
    removed = 0

    for row in range(r1, r2):
        for col in range(c1, c2):
            index = row * cols + col
            if values[index] > 0:
                removed += 1
                values[index] = 0

    return tuple(values), removed


def move_sort_key(move: ScoredMove) -> tuple[int, int, int, int, int, int]:
    r1, r2, c1, c2, apples, area = move
    return apples, area, r2 - r1, c2 - c1, -r1, -c1


def sorted_moves(state: BoardState, rows: int, cols: int) -> list[ScoredMove]:
    moves = find_valid_moves(state, rows, cols)
    moves.sort(key=move_sort_key, reverse=True)
    return moves


def remaining_apples(state: BoardState) -> int:
    return sum(1 for value in state if value > 0)


def greedy_search(initial: BoardState, rows: int, cols: int) -> SearchResult:
    started = time.monotonic()
    state = initial
    score = 0
    moves: list[Move] = []

    while True:
        candidates = sorted_moves(state, rows, cols)
        if not candidates:
            break

        move = candidates[0]
        state, removed = apply_move(state, cols, move)
        if removed == 0:
            break

        score += removed
        moves.append(move[:4])

    return SearchResult("greedy", score, moves, time.monotonic() - started)


def weighted_move_choice(candidates: list[ScoredMove], rng: random.Random) -> ScoredMove:
    pool_size = min(len(candidates), rng.randint(4, 18))
    pool = candidates[:pool_size]
    total_weight = sum(max(1, move[4]) ** 3 for move in pool)
    pick = rng.uniform(0, total_weight)
    running = 0.0

    for move in pool:
        running += max(1, move[4]) ** 3
        if running >= pick:
            return move

    return pool[-1]


def random_restart_search(
    initial: BoardState,
    rows: int,
    cols: int,
    time_limit: float,
    seed: int | None,
) -> SearchResult:
    started = time.monotonic()
    deadline = started + max(0.0, time_limit)
    rng = random.Random(seed)
    best_score = -1
    best_moves: list[Move] = []
    run_index = 0
    total_apples = remaining_apples(initial)

    while True:
        state = initial
        score = 0
        moves: list[Move] = []

        while True:
            candidates = sorted_moves(state, rows, cols)
            if not candidates:
                break

            move = candidates[0] if run_index == 0 else weighted_move_choice(candidates, rng)
            state, removed = apply_move(state, cols, move)
            if removed == 0:
                break

            score += removed
            moves.append(move[:4])

        if score > best_score:
            best_score = score
            best_moves = moves
            if best_score == total_apples:
                break

        run_index += 1
        if time.monotonic() >= deadline:
            break

    return SearchResult("random", max(0, best_score), best_moves, time.monotonic() - started)


def beam_rank(score: int, state: BoardState) -> tuple[int, int]:
    return score, -remaining_apples(state)


def beam_search(
    initial: BoardState,
    rows: int,
    cols: int,
    time_limit: float,
    beam_width: int,
    moves_per_state: int,
) -> SearchResult:
    started = time.monotonic()
    deadline = started + max(0.0, time_limit)
    total_apples = remaining_apples(initial)
    best_score = 0
    best_moves: list[Move] = []
    frontier: list[tuple[BoardState, int, list[Move]]] = [(initial, 0, [])]
    seen_best_score: dict[BoardState, int] = {initial: 0}

    while frontier and time.monotonic() < deadline:
        next_by_state: dict[BoardState, tuple[int, list[Move]]] = {}

        for state, score, moves in frontier:
            if time.monotonic() >= deadline:
                break

            candidates = sorted_moves(state, rows, cols)[:moves_per_state]
            for move in candidates:
                next_state, removed = apply_move(state, cols, move)
                if removed == 0:
                    continue

                next_score = score + removed
                if next_score <= seen_best_score.get(next_state, -1):
                    continue

                next_moves = [*moves, move[:4]]
                seen_best_score[next_state] = next_score
                current = next_by_state.get(next_state)
                if current is None or next_score > current[0]:
                    next_by_state[next_state] = (next_score, next_moves)

                if next_score > best_score:
                    best_score = next_score
                    best_moves = next_moves
                    if best_score == total_apples:
                        return SearchResult(
                            "beam",
                            best_score,
                            best_moves,
                            time.monotonic() - started,
                        )

        ranked = sorted(
            (
                (state, score, moves)
                for state, (score, moves) in next_by_state.items()
            ),
            key=lambda item: beam_rank(item[1], item[0]),
            reverse=True,
        )
        frontier = ranked[:beam_width]

    return SearchResult("beam", best_score, best_moves, time.monotonic() - started)


def depth_first_search(
    initial: BoardState,
    rows: int,
    cols: int,
    time_limit: float,
    branch_limit: int,
) -> SearchResult:
    started = time.monotonic()
    deadline = started + max(0.0, time_limit)
    total_apples = remaining_apples(initial)
    best_score = 0
    best_moves: list[Move] = []
    seen_best_score: dict[BoardState, int] = {initial: 0}
    stack: list[tuple[BoardState, int, list[Move]]] = [(initial, 0, [])]

    while stack and time.monotonic() < deadline:
        state, score, moves = stack.pop()
        if score + remaining_apples(state) <= best_score:
            continue

        candidates = sorted_moves(state, rows, cols)[:branch_limit]
        if not candidates and score > best_score:
            best_score = score
            best_moves = moves
            continue

        for move in reversed(candidates):
            next_state, removed = apply_move(state, cols, move)
            if removed == 0:
                continue

            next_score = score + removed
            if next_score <= seen_best_score.get(next_state, -1):
                continue

            next_moves = [*moves, move[:4]]
            seen_best_score[next_state] = next_score
            if next_score > best_score:
                best_score = next_score
                best_moves = next_moves
                if best_score == total_apples:
                    stack.clear()
                    break

            stack.append((next_state, next_score, next_moves))

    return SearchResult("dfs", best_score, best_moves, time.monotonic() - started)


def solve_with_searches(
    board: list[list[int]],
    algorithm: str,
    time_limit: float,
    seed: int | None,
    beam_width: int,
    moves_per_state: int,
    branch_limit: int,
) -> list[SearchResult]:
    initial, rows, cols = flatten_board(board)
    if remaining_apples(initial) == 0:
        return [SearchResult(algorithm, 0, [], 0.0)]

    if algorithm == "greedy":
        return [greedy_search(initial, rows, cols)]
    if algorithm == "random":
        return [random_restart_search(initial, rows, cols, time_limit, seed)]
    if algorithm == "beam":
        return [beam_search(initial, rows, cols, time_limit, beam_width, moves_per_state)]
    if algorithm == "dfs":
        return [depth_first_search(initial, rows, cols, time_limit, branch_limit)]

    started = time.monotonic()
    deadline = started + max(0.0, time_limit)
    results = [greedy_search(initial, rows, cols)]

    remaining = max(0.0, deadline - time.monotonic())
    beam_time = remaining * 0.45
    if beam_time > 0:
        results.append(
            beam_search(initial, rows, cols, beam_time, beam_width, moves_per_state)
        )

    remaining = max(0.0, deadline - time.monotonic())
    dfs_time = remaining * 0.45
    if dfs_time > 0:
        results.append(depth_first_search(initial, rows, cols, dfs_time, branch_limit))

    remaining = max(0.0, deadline - time.monotonic())
    if remaining > 0:
        results.append(random_restart_search(initial, rows, cols, remaining, seed))

    return results


def print_board(board: list[list[int]]) -> None:
    for row in board:
        print(" ".join(str(value) for value in row))


def print_move_sequence(moves: list[Move]) -> None:
    if not moves:
        print("No moves found.")
        return

    print("Moves, using 1-based inclusive rows and columns:")
    for index, (r1, r2, c1, c2) in enumerate(moves, start=1):
        print(f"{index:02d}. rows {r1 + 1}-{r2}, cols {c1 + 1}-{c2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure search algorithms against a Fruit Box board and keep the "
            "sequence that removes the most apples."
        )
    )
    parser.add_argument(
        "board_json",
        type=Path,
        help="Path to a board JSON file containing a 2D array or {'board': [...]} object.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("all", "greedy", "random", "beam", "dfs"),
        default="all",
        help="Search strategy to run.",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="Total seconds to spend searching. Greedy ignores this.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used by the random-restart search.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=120,
        help="How many partial board states beam search keeps at each depth.",
    )
    parser.add_argument(
        "--moves-per-state",
        type=int,
        default=60,
        help="Maximum candidate moves expanded from each state in beam search.",
    )
    parser.add_argument(
        "--branch-limit",
        type=int,
        default=45,
        help="Maximum candidate moves expanded from each state in DFS search.",
    )
    parser.add_argument(
        "--print-board",
        action="store_true",
        help="Print the loaded board before solving.",
    )
    parser.add_argument(
        "--print-moves",
        action="store_true",
        help="Print the best move sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = load_board_json(args.board_json.resolve())
    initial, rows, cols = flatten_board(board)
    total_apples = remaining_apples(initial)

    print(f"Loaded {rows} x {cols} board with {total_apples} apples.")
    if args.print_board:
        print_board(board)

    results = solve_with_searches(
        board,
        algorithm=args.algorithm,
        time_limit=args.time,
        seed=args.seed,
        beam_width=max(1, args.beam_width),
        moves_per_state=max(1, args.moves_per_state),
        branch_limit=max(1, args.branch_limit),
    )

    for result in results:
        print(
            f"{result.algorithm}: removed {result.score}/{total_apples} apples "
            f"in {len(result.moves)} moves ({result.elapsed:.2f}s)"
        )

    best = max(results, key=lambda result: result.score)
    print(
        f"\nBest search: {best.algorithm} removed {best.score}/{total_apples} apples "
        f"in {len(best.moves)} moves."
    )

    if args.print_moves:
        print_move_sequence(best.moves)


if __name__ == "__main__":
    main()
