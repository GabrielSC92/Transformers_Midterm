#!/usr/bin/env python3
"""
Generate (FEN, UCI) training data using the engine's best move instead of human moves.
Data quality matches model ambition: train to imitate the best available play.

Supports:
  1. Local Stockfish (python-chess) — no API key, no rate limits. Recommended.
  2. Assignment EnginePlayer (RapidAPI) — use if you don't have Stockfish installed.

Output: JSONL file (one {"fen": "...", "uci": "..."} per line) for use in train.ipynb.

In train.ipynb, replace the "Extract (FEN, UCI) pairs" cell with loading from this file:
  from model import MOVE_TO_IDX
  positions = []
  with open("engine_data.jsonl") as f:
      for line in f:
          d = json.loads(line)
          if d["uci"] in MOVE_TO_IDX:
              positions.append((d["fen"], d["uci"]))
  # then run train/val split and rest of notebook as before

Usage:
  # Local Stockfish (install from https://stockfishchess.org or package manager)
  python generate_engine_data.py --mode fens --engine local --target 500000 --out engine_data.jsonl

  # FENs from dataset, engine from RapidAPI (set RAPIDAPI_KEY)
  python generate_engine_data.py --mode fens --engine api --target 100000 --out engine_data.jsonl

  # Run engine vs random games and parse log (EnginePlayer only)
  python generate_engine_data.py --mode games --engine api --games 5000 --out engine_data.jsonl --log engine_log.txt
"""

import argparse
import json
import os
import sys
import threading
import time
from multiprocessing import Process, Queue
from pathlib import Path

import chess
from tqdm import tqdm

# Optional: chess_tournament only needed for EnginePlayer / game-log mode
def _optional_chess_tournament():
    try:
        from chess_tournament import EnginePlayer, Game, RandomPlayer  # pyright: ignore[reportMissingImports]
        return EnginePlayer, Game, RandomPlayer
    except ImportError:
        return None, None, None


def get_engine_move_local(fen: str, engine_path: str, time_limit: float = 0.02, depth: int | None = None, engine=None):
    """Return UCI best move. If engine is provided, reuse it; else start/quit (for single call)."""
    import chess.engine
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    try:
        limit = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(time=time_limit)
        if engine is not None:
            result = engine.play(board, limit)
            return result.move.uci() if result.move else None
        eng = chess.engine.SimpleEngine.popen_uci(engine_path)
        result = eng.play(board, limit)
        eng.quit()
        return result.move.uci() if result.move else None
    except Exception:
        return None


def _worker_process(
    input_queue: Queue,
    output_queue: Queue,
    engine_path: str,
    engine_time: float,
    engine_depth: int | None,
    threads: int,
) -> None:
    """Run in a separate process: consume FENs from input_queue, run Stockfish, put (fen, uci) in output_queue."""
    import chess.engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        try:
            engine.configure({"Threads": threads})
        except Exception:
            pass
        limit = chess.engine.Limit(depth=engine_depth) if engine_depth else chess.engine.Limit(time=engine_time)
        while True:
            fen = input_queue.get()
            if fen is None:
                break
            try:
                board = chess.Board(fen)
                if board.is_game_over():
                    output_queue.put((fen, None))
                    continue
                result = engine.play(board, limit)
                uci = result.move.uci() if result.move else None
                output_queue.put((fen, uci))
            except Exception:
                output_queue.put((fen, None))
        engine.quit()
    finally:
        output_queue.put(None)  # signal this worker is done


def get_engine_move_api(fen: str, engine_player) -> str | None:
    """Return UCI best move using assignment EnginePlayer (RapidAPI)."""
    return engine_player.get_move(fen)


def stream_fens_from_dataset(target: int, min_elo: int = 0):
    """Stream FENs from angeluriot/chess_games by replaying games. Yields (fen) until target reached."""
    from datasets import load_dataset
    count = 0
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    for row in ds:
        if count >= target:
            return
        white_elo = row.get("white_elo") or 0
        black_elo = row.get("black_elo") or 0
        try:
            w, b = int(white_elo or 0), int(black_elo or 0)
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
            fen = board.fen()
            if board.is_game_over():
                break
            try:
                move = chess.Move.from_uci(uci_str)
                if move not in board.legal_moves:
                    break
            except (ValueError, chess.InvalidMoveError):
                break
            count += 1
            yield fen
            board.push(move)


def run_mode_fens(args, out_file):
    """Stream FENs, get engine move for each, and write immediately so interrupt leaves valid partial file."""
    import chess.engine
    written = 0
    # Engine setup
    if args.engine == "local":
        engine_path = args.stockfish_path or "stockfish"
        if not Path(engine_path).exists() and engine_path == "stockfish":
            try:
                import shutil
                if shutil.which("stockfish"):
                    engine_path = shutil.which("stockfish")
            except Exception:
                pass
        # Resolve to absolute path so Windows can find the exe
        engine_path = str(Path(engine_path).resolve())
        p = Path(engine_path)
        if not p.is_file():
            # Try same dir or parent dir; only accept executable files (never a directory)
            exe_names = [
                "stockfish-windows-x86-64-avx2.exe",
                "Stockfish.exe",
                "stockfish.exe",
                "stockfish",
            ]
            found = None
            for folder in ([p, p.parent, p.parent.parent] if p.is_dir() else [p.parent, p.parent.parent]):
                for name in exe_names:
                    alt = folder / name
                    if alt.exists() and alt.is_file():
                        found = alt.resolve()
                        break
                if found is not None:
                    break
            if found is not None:
                engine_path = str(found)
            else:
                print(f"Stockfish not found at: {engine_path}", file=sys.stderr)
                print("Check the path in Explorer and pass --stockfish-path with the exact exe path.", file=sys.stderr)
                sys.exit(1)
        print(f"Using local Stockfish: {engine_path} (time={args.engine_time}s, depth={args.engine_depth}, threads={args.threads}, workers={args.workers})")
        if args.workers <= 1:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            try:
                engine.configure({"Threads": args.threads})
            except Exception:
                pass
            def get_move(fen):
                return get_engine_move_local(fen, engine_path, time_limit=args.engine_time, depth=args.engine_depth, engine=engine)
        else:
            engine = None  # multi-worker path uses processes, no single engine here
    else:
        engine = None
        EnginePlayer, _, _ = _optional_chess_tournament()
        if EnginePlayer is None:
            print("Install chess_exam (pip install -e . in chess_exam) and set RAPIDAPI_KEY for API engine.", file=sys.stderr)
            sys.exit(1)
        key = os.environ.get("RAPIDAPI_KEY", "")
        if not key:
            print("Set RAPIDAPI_KEY for EnginePlayer.", file=sys.stderr)
            sys.exit(1)
        engine_instance = EnginePlayer("Engine", blunder_rate=0.0, ponder_rate=0.0)
        print("Using EnginePlayer (RapidAPI). Throttling to avoid rate limits.")
        def get_move(fen):
            move = get_engine_move_api(fen, engine_instance)
            time.sleep(args.api_delay)
            return move
    try:
        if args.engine == "local" and args.workers > 1:
            # Multi-worker: N Stockfish processes in parallel
            input_queue: Queue = Queue()
            output_queue: Queue = Queue()
            workers_list = [
                Process(
                    target=_worker_process,
                    args=(input_queue, output_queue, engine_path, args.engine_time, args.engine_depth, args.threads),
                )
                for _ in range(args.workers)
            ]
            for w in workers_list:
                w.start()

            def producer():
                if args.fen_file:
                    with open(args.fen_file) as fr:
                        for i, line in enumerate(fr):
                            if i >= args.target:
                                break
                            fen = line.strip()
                            if fen:
                                input_queue.put(fen)
                else:
                    for fen in stream_fens_from_dataset(args.target, min_elo=args.min_elo):
                        input_queue.put(fen)
                for _ in range(args.workers):
                    input_queue.put(None)

            prod_thread = threading.Thread(target=producer)
            prod_thread.start()

            stopped = 0
            with open(out_file, "w") as f:
                pbar = tqdm(desc="Engine moves", total=args.target, unit="pairs")
                while stopped < args.workers:
                    item = output_queue.get()
                    if item is None:
                        stopped += 1
                        continue
                    fen, uci = item
                    if uci:
                        f.write(json.dumps({"fen": fen, "uci": uci}) + "\n")
                        f.flush()
                        written += 1
                        pbar.update(1)
                    if written % 10000 == 0 and written > 0:
                        tqdm.write(f"  Written {written} pairs (Ctrl+C leaves file valid)")
                pbar.close()
            prod_thread.join()
            for w in workers_list:
                w.join()
            print(f"Wrote {written} (FEN, UCI) pairs to {out_file}")
        else:
            with open(out_file, "w") as f:
                if args.fen_file:
                    with open(args.fen_file) as fr:
                        fens = (line.strip() for line in fr if line.strip())
                    for fen in tqdm(fens, desc="Engine moves", total=args.target):
                        if written >= args.target:
                            break
                        move = get_move(fen)
                        if move:
                            f.write(json.dumps({"fen": fen, "uci": move}) + "\n")
                            f.flush()
                            written += 1
                else:
                    pbar = tqdm(desc="Engine moves", total=args.target, unit="pairs")
                    for fen in stream_fens_from_dataset(args.target, min_elo=args.min_elo):
                        move = get_move(fen)
                        if move:
                            f.write(json.dumps({"fen": fen, "uci": move}) + "\n")
                            f.flush()
                            written += 1
                            pbar.update(1)
                        if written % 10000 == 0 and written > 0:
                            tqdm.write(f"  Written {written} pairs (Ctrl+C leaves file valid)")
                    pbar.close()
            print(f"Wrote {written} (FEN, UCI) pairs to {out_file}")
    finally:
        if args.engine == "local" and engine is not None:
            engine.quit()


def run_mode_games(args, out_file):
    """Run engine vs random games, log moves, parse log to extract (FEN, UCI) for engine."""
    EnginePlayer, Game, RandomPlayer = _optional_chess_tournament()
    if EnginePlayer is None or Game is None or RandomPlayer is None:
        print("Install chess_exam (pip install -e . in chess_exam) for game mode.", file=sys.stderr)
        sys.exit(1)
    key = os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        print("Set RAPIDAPI_KEY for EnginePlayer.", file=sys.stderr)
        sys.exit(1)
    engine = EnginePlayer("Engine", blunder_rate=0.0, ponder_rate=0.0)
    random_player = RandomPlayer("Random")
    engine_name = "Engine"
    all_positions = []
    log_stem = Path(args.log_file).stem if args.log_file else "engine_log"
    log_dir = Path(args.log_file).parent if args.log_file else Path("engine_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {args.games} games (Engine vs Random), one log file per game")
    for g in tqdm(range(args.games), desc="Games"):
        per_game_log = str(log_dir / f"{log_stem}_{g:06d}.txt")
        game = Game(engine, random_player, max_half_moves=args.max_half_moves)
        game.play(verbose=False, log_moves=True, log_to_file=per_game_log)
        # Parse this game's log
        try:
            with open(per_game_log) as f:
                for line in f:
                    line = line.strip()
                    if not line or "Game finished" in line or "Fallback" in line:
                        continue
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 5:
                        continue
                    player, fen, move = parts[1], parts[3], parts[4]
                    if player != engine_name or not move:
                        continue
                    board = chess.Board(fen)
                    try:
                        uci = board.parse_san(move).uci()
                    except Exception:
                        try:
                            chess.Move.from_uci(move)
                            uci = move
                        except Exception:
                            continue
                    all_positions.append((fen, uci))
        except FileNotFoundError:
            pass
    with open(out_file, "w") as f:
        for fen, uci in all_positions:
            f.write(json.dumps({"fen": fen, "uci": uci}) + "\n")
    print(f"Extracted {len(all_positions)} engine (FEN, UCI) pairs → {out_file}")


def main():
    p = argparse.ArgumentParser(description="Generate (FEN, UCI) training data from engine best moves.")
    p.add_argument("--mode", choices=["fens", "games"], default="fens",
                   help="fens: get FENs then query engine; games: run engine vs random and parse log")
    p.add_argument("--engine", choices=["local", "api"], default="local",
                   help="local=Stockfish binary; api=assignment EnginePlayer (RapidAPI)")
    p.add_argument("--out", default="engine_data.jsonl", help="Output JSONL file")
    p.add_argument("--target", type=int, default=500_000, help="Target number of (FEN, UCI) pairs (fens mode)")
    p.add_argument("--fen-file", type=str, default=None, help="File with one FEN per line (optional, else use dataset)")
    p.add_argument("--min-elo", type=int, default=0, help="Min white/black ELO when sampling FENs from dataset")
    p.add_argument("--stockfish-path", type=str, default=None, help="Path to Stockfish binary (default: stockfish in PATH)")
    p.add_argument("--engine-time", type=float, default=0.02, help="Seconds per move for local Stockfish (lower = faster, slightly weaker)")
    p.add_argument("--engine-depth", type=int, default=None, help="If set, use depth limit instead of time (e.g. 5 = fast, 10 = stronger)")
    p.add_argument("--threads", type=int, default=4, help="Stockfish UCI Threads per engine (more = faster per position)")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel Stockfish processes (e.g. 8 for 8 cores)")
    p.add_argument("--api-delay", type=float, default=0.5, help="Delay between API calls (seconds) to avoid rate limit")
    # games mode
    p.add_argument("--games", type=int, default=5000, help="Number of games to run (games mode)")
    p.add_argument("--log-file", type=str, default=None, help="Log file path (games mode)")
    p.add_argument("--max-half-moves", type=int, default=150, help="Max half-moves per game (games mode)")
    args = p.parse_args()

    if args.mode == "games":
        run_mode_games(args, args.out)
    else:
        run_mode_fens(args, args.out)


if __name__ == "__main__":
    main()
