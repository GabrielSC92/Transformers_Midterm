"""Generate (FEN, UCI) training data from Stockfish best moves."""

import json
import os
import sys
import threading
from multiprocessing import Process, Queue
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm

# ── Configuration (tweak these before each run) ──────────────────────────────
OUTPUT_FILE = "engine_data.jsonl"
TARGET_POSITIONS = 500_000
MIN_ELO = 0

STOCKFISH_PATH = "stockfish"  # or full path to exe
ENGINE_DEPTH = 10
THREADS_PER_ENGINE = 3
HASH_MB = 128
NUM_WORKERS = 8
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_stockfish(path: str) -> str:
    """Find the Stockfish executable, searching common locations on Windows."""
    import shutil

    p = Path(path).resolve()
    if p.is_file():
        return str(p)

    found = shutil.which("stockfish")
    if found:
        return found

    if sys.platform == "win32":
        exe_names = [
            "stockfish-windows-x86-64-avx2.exe",
            "Stockfish.exe",
            "stockfish.exe",
        ]
        search_dirs = [Path.cwd(), p.parent, p.parent.parent]
        downloads = Path(os.path.expanduser("~")) / "Downloads"
        search_dirs.extend(downloads.iterdir() if downloads.is_dir() else [])

        for folder in search_dirs:
            if not folder.is_dir():
                continue
            for name in exe_names:
                candidate = folder / name
                if candidate.is_file():
                    return str(candidate.resolve())
            for sub in folder.iterdir():
                if sub.is_dir():
                    for name in exe_names:
                        candidate = sub / name
                        if candidate.is_file():
                            return str(candidate.resolve())

    print(f"Stockfish not found at: {path}", file=sys.stderr)
    sys.exit(1)


def _worker(input_q: Queue, output_q: Queue, engine_path: str) -> None:
    """Stockfish worker process: reads FENs, writes (fen, uci) pairs."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        engine.configure({"Threads": THREADS_PER_ENGINE, "Hash": HASH_MB, "Skill Level": 20})
    except Exception:
        pass
    limit = chess.engine.Limit(depth=ENGINE_DEPTH)

    while True:
        fen = input_q.get()
        if fen is None:
            break
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                continue
            result = engine.play(board, limit)
            if result.move:
                output_q.put((fen, result.move.uci()))
        except Exception:
            pass

    engine.quit()
    output_q.put(None)


def stream_fens(target: int, min_elo: int = 0):
    """Yield FENs from angeluriot/chess_games by replaying moves."""
    from datasets import load_dataset

    count = 0
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    for row in ds:
        if count >= target:
            return
        try:
            w, b = int(row.get("white_elo") or 0), int(row.get("black_elo") or 0)
            if w < min_elo or b < min_elo:
                continue
        except (TypeError, ValueError):
            continue
        uci_moves = row.get("moves_uci")
        if not uci_moves:
            continue
        board = chess.Board()
        for uci_str in uci_moves:
            if count >= target:
                return
            if board.is_game_over():
                break
            try:
                move = chess.Move.from_uci(uci_str)
                if move not in board.legal_moves:
                    break
            except (ValueError, chess.InvalidMoveError):
                break
            yield board.fen()
            board.push(move)
            count += 1


def main() -> None:
    engine_path = _resolve_stockfish(STOCKFISH_PATH)
    print(f"Stockfish: {engine_path}")
    print(f"depth={ENGINE_DEPTH}, threads={THREADS_PER_ENGINE}, hash={HASH_MB}MB, workers={NUM_WORKERS}")
    print(f"Target: {TARGET_POSITIONS} positions → {OUTPUT_FILE}")

    input_q: Queue = Queue()
    output_q: Queue = Queue()

    workers = [
        Process(target=_worker, args=(input_q, output_q, engine_path))
        for _ in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    def producer():
        for fen in stream_fens(TARGET_POSITIONS, min_elo=MIN_ELO):
            input_q.put(fen)
        for _ in range(NUM_WORKERS):
            input_q.put(None)

    prod = threading.Thread(target=producer)
    prod.start()

    written = 0
    stopped = 0
    with open(OUTPUT_FILE, "w") as f:
        pbar = tqdm(desc="Engine moves", total=TARGET_POSITIONS, unit="pos")
        while stopped < NUM_WORKERS:
            item = output_q.get()
            if item is None:
                stopped += 1
                continue
            fen, uci = item
            f.write(json.dumps({"fen": fen, "uci": uci}) + "\n")
            f.flush()
            written += 1
            pbar.update(1)
        pbar.close()

    prod.join()
    for w in workers:
        w.join()
    print(f"Done — wrote {written} (FEN, UCI) pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
