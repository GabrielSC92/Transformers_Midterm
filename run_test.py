#!/usr/bin/env python3
"""
Evaluate TransformerPlayer against assignment baseline opponents.

Supports: random, engine-strong, engine-weak, local-strong, local-weak, lm, smol, sayed.
Use --opponent to pick one, or --all to run every available opponent in sequence.

local-strong/local-weak use a LOCAL Stockfish binary (no API key needed, much faster).
They replicate the tournament EnginePlayer blunder/ponder behavior.

Before any games, runs the same validation the tournament uses:
  1. TransformerPlayer() instantiated with zero arguments
  2. get_move called on the tournament test FEN
  3. UCI format verified against the tournament regex
  4. Total time checked against the 60s limit

Examples:
  python run_test.py                                  # 200 games vs Random (default)
  python run_test.py --opponent engine-strong --games 50
  python run_test.py --all --games 100
  python run_test.py --all --exclude lm smol --games 20
  python run_test.py --watch --opponent random --local-dir .   # one game with live ASCII board
"""

import argparse
import os
import random
import re
import sys
import time
from collections import Counter
from types import SimpleNamespace
from typing import Optional

import chess

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:
    pass

from chess_tournament import Game, RandomPlayer
from player import TransformerPlayer

OPPONENT_CHOICES = ["random", "engine-strong", "engine-weak", "local-strong", "local-weak", "lm", "smol", "sayed"]
MODEL_CHOICES = ["recurrent", "sayed"]
COLOR_CHOICES = ["white", "black", "alternating"]
TP_NAME = "RecurrentTransformer"
SAYED_NAME = "Sayed_player"

TOURNAMENT_TEST_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)
TOURNAMENT_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Local Stockfish player (no API needed)
# ---------------------------------------------------------------------------

class LocalEnginePlayer:
    """Local Stockfish opponent that mimics the tournament EnginePlayer behavior.

    blunder_rate: probability of playing a random legal move
    ponder_rate:  probability of playing the second-best move
    remaining:    probability of playing the best move
    """

    def __init__(self, name: str, blunder_rate: float = 0.0, ponder_rate: float = 0.0,
                 depth: int = 12, threads: int = 2, hash_mb: int = 64):
        self.name = name
        self.blunder_rate = blunder_rate
        self.ponder_rate = ponder_rate
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine = None
        self._engine_path = self._find_stockfish()

    @staticmethod
    def _find_stockfish() -> str:
        import shutil
        from pathlib import Path

        candidates = ["stockfish"]
        if sys.platform == "win32":
            candidates = [
                "stockfish-windows-x86-64-avx2.exe",
                "Stockfish.exe",
                "stockfish.exe",
                "stockfish",
            ]

        for name in candidates:
            found = shutil.which(name)
            if found:
                return found

        if sys.platform == "win32":
            from pathlib import Path
            fallback = Path(os.path.expanduser("~")) / "Downloads" / "stockfish-windows-x86-64-avx2" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
            if fallback.is_file():
                return str(fallback.resolve())

        raise FileNotFoundError(
            "Stockfish not found. Install it or pass --stockfish-path. "
            "Download from https://stockfishchess.org"
        )

    def _get_engine(self):
        if self._engine is None:
            import chess.engine
            self._engine = chess.engine.SimpleEngine.popen_uci(self._engine_path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self._engine

    def get_move(self, fen: str) -> Optional[str]:
        import chess.engine as ce
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return None

        r = random.random()

        # Blunder: random legal move
        if r < self.blunder_rate:
            return random.choice(legal).uci()

        engine = self._get_engine()
        limit = ce.Limit(depth=self.depth)

        # Get best + second-best via multipv
        if self.ponder_rate > 0 and r < self.blunder_rate + self.ponder_rate:
            try:
                infos = engine.analyse(board, limit, multipv=2)
                if len(infos) >= 2 and infos[1].get("pv"):
                    return infos[1]["pv"][0].uci()  # type: ignore[index]
                elif infos and infos[0].get("pv"):
                    return infos[0]["pv"][0].uci()  # type: ignore[index]
            except Exception:
                pass
            return random.choice(legal).uci()

        # Best move
        try:
            result = engine.play(board, limit)
            if result.move:
                return result.move.uci()
        except Exception:
            pass
        return random.choice(legal).uci()

    def close(self):
        if self._engine:
            self._engine.quit()
            self._engine = None

    def __del__(self):
        self.close()

# ---------------------------------------------------------------------------
# Tournament-style validation (runs automatically before every test)
# ---------------------------------------------------------------------------


def run_tournament_validation(
        local_dir: Optional[str] = None) -> TransformerPlayer:
    """Replicate the tournament's validation checks.

    Returns the instantiated player so it can be reused for games.
    Exits with an error if any check fails.
    If local_dir is set, loads from that directory (config + tokenizer + model.pt/best_model.pt)
    instead of HuggingFace; use this to test your local best model before pushing.
    """
    print("=" * 50)
    print("TOURNAMENT VALIDATION")
    print("=" * 50)

    t0 = time.time()

    # 1. Instantiation (no-arg = HuggingFace; with local_dir = local model)
    if local_dir:
        print(
            f"  [1/3] Instantiating TransformerPlayer(local_dir={local_dir!r}) ... ",
            end="",
            flush=True)
        try:
            tp = TransformerPlayer(local_dir=local_dir)
        except Exception as e:
            print("FAIL")
            print(f"\n  TransformerPlayer(local_dir=...) raised: {e}")
            sys.exit(1)
        print("OK (local model)")
    else:
        print(
            "  [1/3] Instantiating TransformerPlayer() with no arguments ... ",
            end="",
            flush=True)
        try:
            tp = TransformerPlayer()
        except Exception as e:
            print("FAIL")
            print(f"\n  TransformerPlayer() raised: {e}")
            print(
                "  The tournament calls cls() with zero args — this would be rejected."
            )
            sys.exit(1)
    init_time = time.time() - t0
    if not local_dir:
        print(f"OK ({init_time:.1f}s)")

    # 2. get_move on the tournament's test FEN
    print(f"  [2/3] Calling get_move(test_fen) ... ", end="", flush=True)
    try:
        move = tp.get_move(TOURNAMENT_TEST_FEN)
    except Exception as e:
        print("FAIL")
        print(f"\n  get_move raised: {e}")
        sys.exit(1)
    move_time = time.time() - t0 - init_time
    print(f"OK  -> '{move}' ({move_time:.2f}s)")

    # 3. UCI format check
    print(f"  [3/3] Checking UCI format ... ", end="", flush=True)
    if move is None:
        print("FAIL")
        print("\n  get_move returned None on a position with legal moves.")
        sys.exit(1)
    if not UCI_RE.match(move.strip()):
        print("FAIL")
        print(
            f"\n  Move '{move}' does not match tournament regex {UCI_RE.pattern}"
        )
        sys.exit(1)
    print("OK")

    total = time.time() - t0
    if total > TOURNAMENT_TIMEOUT:
        print(
            f"\n  WARNING: Validation took {total:.1f}s (tournament limit: {TOURNAMENT_TIMEOUT}s)"
        )
        print("  Your player would be TIMED OUT in the real tournament!")
        sys.exit(1)
    elif total > TOURNAMENT_TIMEOUT * 0.5:
        print(
            f"\n  WARNING: Validation took {total:.1f}s — approaching the {TOURNAMENT_TIMEOUT}s limit."
        )
    else:
        print(
            f"\n  All checks passed in {total:.1f}s (limit: {TOURNAMENT_TIMEOUT}s)"
        )
    print("=" * 50)

    return tp


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


def _check_rapidapi_key() -> bool:
    return bool(os.environ.get("RAPIDAPI_KEY", ""))


def _check_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN", ""))


def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_opponent_available(name: str) -> tuple[bool, str]:
    """Return (available, reason) for a given opponent."""
    if name == "random":
        return True, ""
    if name in ("engine-strong", "engine-weak"):
        if _check_rapidapi_key():
            return True, ""
        return False, "RAPIDAPI_KEY env var not set"
    if name in ("local-strong", "local-weak"):
        try:
            LocalEnginePlayer._find_stockfish()
            return True, ""
        except FileNotFoundError as e:
            return False, str(e)
    if name == "lm":
        if not _check_gpu():
            return False, "No GPU available (LMPlayer needs CUDA)"
        return True, ""
    if name == "smol":
        if not _check_hf_token():
            return False, "HF_TOKEN env var not set"
        return True, ""
    if name == "sayed":
        if not _check_gpu():
            return False, "No GPU available (SayedPlayer needs CUDA)"
        return True, ""
    return False, f"Unknown opponent: {name}"


# ---------------------------------------------------------------------------
# Opponent factory
# ---------------------------------------------------------------------------


def make_opponent(name: str, stockfish_path: str | None = None):
    """Instantiate an opponent player by name."""
    if name == "random":
        return RandomPlayer("Random")

    if name == "engine-strong":
        from chess_tournament import EnginePlayer
        return EnginePlayer("Stockfish-Strong",
                            blunder_rate=0.0,
                            ponder_rate=0.05)
        
    if name == "sayed":
        import importlib.util
        sayed_path = os.path.join(_REPO_ROOT, "INFOMTALC-chess", "player.py")
        spec = importlib.util.spec_from_file_location("infomtalc_chess_player", sayed_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"SayedPlayer module not found: {sayed_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TransformerPlayer("Sayed_player")

    if name == "engine-weak":
        from chess_tournament import EnginePlayer
        return EnginePlayer("Stockfish-Weak",
                            blunder_rate=0.25,
                            ponder_rate=0.65)

    if name == "local-strong":
        p = LocalEnginePlayer("LocalStockfish-Strong",
                              blunder_rate=0.0, ponder_rate=0.05, depth=12)
        if stockfish_path:
            p._engine_path = stockfish_path
        return p

    if name == "local-weak":
        p = LocalEnginePlayer("LocalStockfish-Weak",
                              blunder_rate=0.25, ponder_rate=0.65, depth=12)
        if stockfish_path:
            p._engine_path = stockfish_path
        return p

    if name == "lm":
        from chess_tournament import LMPlayer
        return LMPlayer("Mistral", quantization="4bit")

    if name == "smol":
        from chess_tournament import SmolPlayer
        return SmolPlayer("Smol")

    raise ValueError(f"Unknown opponent: {name}")


def make_main_player(model: str, local_dir: Optional[str] = None):
    """Create the main player (model under test). Returns (player, display_name)."""
    if model == "recurrent":
        tp = run_tournament_validation(local_dir=local_dir)
        return tp, TP_NAME
    if model == "sayed":
        import importlib.util
        sayed_path = os.path.join(_REPO_ROOT, "INFOMTALC-chess", "player.py")
        spec = importlib.util.spec_from_file_location("infomtalc_chess_player", sayed_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"SayedPlayer module not found: {sayed_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TransformerPlayer(SAYED_NAME), SAYED_NAME
    raise ValueError(f"Unknown model: {model}")


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------


def run_matchup(tp,
                opponent,
                num_games: int,
                max_half_moves: int,
                white_only: bool = False,
                black_only: bool = False,
                tp_name: str = TP_NAME) -> dict:
    """Play num_games. If white_only, tp always white; if black_only, tp always black; else alternating."""
    results = Counter()
    tp_wins = 0
    opp_wins = 0
    draws = 0
    fallbacks_tp = 0
    fallbacks_opp = 0
    opp_name = opponent.name

    t0 = time.time()
    for i in range(num_games):
        if white_only:
            white, black = tp, opponent
        elif black_only:
            white, black = opponent, tp
        else:
            white = tp if i % 2 == 0 else opponent
            black = opponent if i % 2 == 0 else tp
        game = Game(white, black, max_half_moves=max_half_moves)
        outcome, scores, fallbacks = game.play(verbose=False, force_colors=(white, black))

        results[outcome] += 1
        fallbacks_tp += fallbacks.get(tp_name, 0)
        fallbacks_opp += fallbacks.get(opp_name, 0)

        if white_only:
            if outcome == "1-0":
                tp_wins += 1
            elif outcome == "0-1":
                opp_wins += 1
            else:
                draws += 1
        elif black_only:
            if outcome == "0-1":
                tp_wins += 1
            elif outcome == "1-0":
                opp_wins += 1
            else:
                draws += 1
        else:
            if outcome == "1-0":
                tp_wins += 1 if i % 2 == 0 else 0
                opp_wins += 1 if i % 2 == 1 else 0
            elif outcome == "0-1":
                opp_wins += 1 if i % 2 == 0 else 0
                tp_wins += 1 if i % 2 == 1 else 0
            else:
                draws += 1

        # Progress: every 10 games (so engine matchups don't look stuck), or every 50 for fast opponents
        step = 10 if num_games <= 100 else 50
        if (i + 1) % step == 0 or (i + 1) == num_games:
            print(f"  ... {i + 1}/{num_games} games done", flush=True)

    elapsed = time.time() - t0

    return {
        "opponent": opp_name,
        "games": num_games,
        "tp_wins": tp_wins,
        "opp_wins": opp_wins,
        "draws": draws,
        "fallbacks_tp": fallbacks_tp,
        "fallbacks_opp": fallbacks_opp,
        "results": results,
        "elapsed": elapsed,
    }


def _clear_screen():
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")


def draw_board(board: chess.Board,
               white_name: str,
               black_name: str,
               last_move_uci: Optional[str] = None) -> None:
    """Print a small ASCII chessboard (white on bottom)."""
    # Build 8x8 grid: rank 8 at top, rank 1 at bottom (white pieces at bottom)
    lines = []
    lines.append(f"  {white_name} (white) vs {black_name} (black)")
    lines.append("")
    lines.append("    a b c d e f g h")
    for rank in range(7, -1, -1):
        row = [f" {rank + 1} "]
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                row.append(piece.symbol())
            else:
                row.append(".")
        row.append(f" {rank + 1}")
        lines.append("".join(row))
    lines.append("    a b c d e f g h")
    if last_move_uci:
        lines.append(f"  Last move: {last_move_uci}")
    lines.append(
        f"  Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print("\n".join(lines))


def run_watch_game(tp,
                   opponent,
                   opponent_name: str,
                   max_half_moves: int = 200,
                   clear_screen: bool = True,
                   tp_name: str = TP_NAME,
                   tp_plays_white: bool = True) -> str:
    """Run one game with live ASCII board. tp_plays_white: main model plays white if True. Returns outcome (e.g. '1-0')."""
    board = chess.Board()
    if tp_plays_white:
        white, black = tp, opponent
        white_name, black_name = tp_name, opponent_name
    else:
        white, black = opponent, tp
        white_name, black_name = opponent_name, tp_name
    move_count = 0
    last_move_uci: Optional[str] = None

    if clear_screen:
        _clear_screen()
    print(
        f"  {white_name} (White) vs {black_name} (Black) — watch mode (one game)"
    )
    print("  Press Ctrl+C to stop.\n")
    draw_board(board, white_name, black_name, last_move_uci)

    while not board.is_game_over() and move_count < max_half_moves:
        current = white if board.turn == chess.WHITE else black
        fen = board.fen()
        move_uci = current.get_move(fen)
        move = None
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci.strip())
            except ValueError:
                pass
        if move is None or move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
            move_uci = move.uci()
            whose = black_name if board.turn == chess.BLACK else white_name
            print(
                f"  [{whose} returned no/invalid move — using random legal move]",
                flush=True)
        board.push(move)
        move_count += 1
        last_move_uci = move_uci

        if clear_screen:
            _clear_screen()
        print(
            f"  {white_name} (White) vs {black_name} (Black) — move {move_count}\n"
        )
        draw_board(board, white_name, black_name, last_move_uci)
        print(flush=True)

    outcome = board.result()
    print(f"\n  Result: {outcome}  (after {move_count} half-moves)")
    if outcome == "1-0":
        print(f"  → {white_name} (White) wins.")
    elif outcome == "0-1":
        print(f"  → {black_name} (Black) wins.")
    elif outcome == "1/2-1/2":
        print("  → Draw.")
    return outcome


def print_matchup_result(r: dict, tp_name: str = TP_NAME):
    """Print detailed results for a single matchup."""
    n = r["games"]
    print(f"\nResults over {n} games vs {r['opponent']}:")
    for key in ["1-0", "0-1", "1/2-1/2", "*"]:
        count = r["results"].get(key, 0)
        if count > 0:
            print(f"  {key}: {count} ({100 * count / n:.0f}%)")
    print(f"\n  {tp_name} wins: {r['tp_wins']}")
    print(f"  {r['opponent']} wins:  {r['opp_wins']}")
    print(f"  Draws:            {r['draws']}")
    print(f"  Fallbacks (ours): {r['fallbacks_tp']}")
    print(f"  Fallbacks (opp):  {r['fallbacks_opp']}")
    decided = r["tp_wins"] + r["opp_wins"]
    if decided > 0:
        print(f"  Win rate (excl. draws): {100 * r['tp_wins'] / decided:.1f}%")
    print(f"  Time: {r['elapsed']:.1f}s ({r['elapsed'] / n:.2f}s/game)")


def print_summary_table(all_results: list[dict]):
    """Print a combined summary table across all matchups."""
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    header = f"{'Opponent':<20} {'Games':>5} {'Wins':>5} {'Losses':>6} {'Draws':>6} {'FB':>4} {'Win%':>7}"
    print(header)
    print("-" * 75)
    for r in all_results:
        decided = r["tp_wins"] + r["opp_wins"]
        win_pct = f"{100 * r['tp_wins'] / decided:.1f}%" if decided > 0 else "N/A"
        row = f"{r['opponent']:<20} {r['games']:>5} {r['tp_wins']:>5} {r['opp_wins']:>6} {r['draws']:>6} {r['fallbacks_tp']:>4} {win_pct:>7}"
        print(row)
    print("=" * 75)


# ---------------------------------------------------------------------------
# Interactive prompts (when run with no arguments)
# ---------------------------------------------------------------------------


def _prompt(text: str, default: str = "") -> str:
    """Prompt and return stripped input or default if empty."""
    out = input(text).strip()
    return out if out else default


def _interactive_args() -> SimpleNamespace:
    """Build args from interactive prompts when user runs script with no arguments."""
    print("Run with no arguments — interactive mode.\n")
    watch = _prompt("Watch one game with live board? (y/n) [n]: ",
                    "n").lower() == "y"
    if watch:
        print("\nModels:", ", ".join(MODEL_CHOICES))
        model = _prompt("Which model? (recurrent / sayed) [recurrent]: ", "recurrent").lower()
        if model not in MODEL_CHOICES:
            model = "recurrent"
        print("\nStarting player: white, black")
        color = _prompt("Model plays (white / black) [white]: ", "white").lower()
        if color not in ("white", "black"):
            color = "white"
        white_only = color == "white"
        print("\nOpponents:", ", ".join(OPPONENT_CHOICES))
        opp = _prompt("Opponent [random]: ", "random").lower()
        if opp not in OPPONENT_CHOICES:
            opp = "random"
        local = _prompt("Use local model (current dir)? (y/n) [y]: ", "y").lower() != "n" if model == "recurrent" else False
        return SimpleNamespace(
            opponent=opp,
            all=False,
            exclude=[],
            games=200,
            max_half_moves=200,
            local_dir="." if local else None,
            white_only=white_only,
            color=color,
            model=model,
            watch=True,
            stockfish_path=None,
        )
    print("\nModels:", ", ".join(MODEL_CHOICES))
    model = _prompt("Which model? (recurrent / sayed) [recurrent]: ", "recurrent").lower()
    if model not in MODEL_CHOICES:
        model = "recurrent"
    print("\nStarting player: white, black, alternating")
    color = _prompt("Model plays (white / black / alternating) [alternating]: ", "alternating").lower()
    if color not in COLOR_CHOICES:
        color = "alternating"
    print("\nOpponents:", ", ".join(OPPONENT_CHOICES) + ", all")
    opp = _prompt("Opponent (or 'all') [random]: ", "random").lower()
    if opp not in OPPONENT_CHOICES and opp != "all":
        opp = "random"
    use_all = opp == "all"
    exclude: list[str] = []
    if use_all:
        ex = _prompt("Exclude any? (comma-separated, e.g. lm,smol) []: ", "")
        if ex:
            exclude = [
                x.strip().lower() for x in ex.split(",")
                if x.strip() in OPPONENT_CHOICES
            ]
    games_s = _prompt("Number of games [200]: ", "200")
    try:
        games = int(games_s) if games_s else 200
    except ValueError:
        games = 200
    local = _prompt("Use local model (current dir)? (y/n) [y]: ",
                    "y").lower() != "n" if model == "recurrent" else False
    return SimpleNamespace(
        opponent=opp if not use_all else "random",
        all=use_all,
        exclude=exclude,
        games=games,
        max_half_moves=200,
        local_dir="." if local else None,
        white_only=(color == "white"),
        color=color,
        model=model,
        watch=False,
        stockfish_path=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TransformerPlayer against baseline opponents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENT_CHOICES,
        default="random",
        help="Opponent to play against (default: random)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run against every available opponent in sequence",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=OPPONENT_CHOICES,
        default=[],
        help="Opponents to skip when using --all (e.g. --exclude lm smol)",
    )
    parser.add_argument("--games",
                        type=int,
                        default=200,
                        help="Games per matchup (default: 200)")
    parser.add_argument("--max-half-moves",
                        type=int,
                        default=200,
                        help="Max half-moves per game (default: 200)")
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="recurrent",
        help="Main player model: recurrent (yours) or sayed (INFOMTALC-chess) (default: recurrent)",
    )
    parser.add_argument(
        "--color",
        choices=COLOR_CHOICES,
        default="alternating",
        help="Starting player: white (model always white), black (model always black), or alternating (default: alternating)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=
        "Load model from DIR (config.json, tokenizer.json, best_model.pt) instead of HuggingFace. Use '.' for repo root. Only for --model recurrent.",
    )
    parser.add_argument(
        "--white-only",
        action="store_true",
        help=
        "Model always plays white (same as --color white). Default: alternating.",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to local Stockfish binary (auto-detected if not set)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=
        "Run a single game and draw a live ASCII board in the terminal (use with --opponent).",
    )
    # If no arguments given, run interactive prompts instead of defaults
    if len(sys.argv) == 1:
        args = _interactive_args()
    else:
        args = parser.parse_args()

    # Derive color options (--white-only overrides --color)
    color = getattr(args, "color", "alternating")
    white_only = getattr(args, "white_only", False) or (color == "white")
    black_only = (color == "black") and not white_only

    # Watch mode: one game with live board, then exit
    if args.watch:
        if args.model == "sayed":
            ok, reason = check_opponent_available("sayed")
            if not ok:
                print(f"Cannot use Sayed model: {reason}")
                sys.exit(1)
        tp, tp_name = make_main_player(args.model, local_dir=getattr(args, "local_dir", None))
        ok, reason = check_opponent_available(args.opponent)
        if not ok:
            print(f"Cannot watch vs {args.opponent}: {reason}")
            sys.exit(1)
        opponent = make_opponent(args.opponent, stockfish_path=getattr(args, "stockfish_path", None))
        run_watch_game(
            tp,
            opponent,
            args.opponent,
            max_half_moves=args.max_half_moves,
            tp_name=tp_name,
            tp_plays_white=white_only or (not black_only),
        )
        return

    opponents = OPPONENT_CHOICES if args.all else [args.opponent]
    opponents = [o for o in opponents if o not in args.exclude]
    if args.model == "sayed":
        opponents = [o for o in opponents if o != "sayed"]

    # Pre-check availability
    available = []
    for name in opponents:
        ok, reason = check_opponent_available(name)
        if ok:
            available.append(name)
        else:
            print(f"[SKIP] {name}: {reason}")

    if not available:
        print("\nNo opponents available. Check API keys / GPU.")
        sys.exit(1)

    if args.model == "sayed":
        ok, reason = check_opponent_available("sayed")
        if not ok:
            print(f"\nSayed model requires GPU: {reason}")
            sys.exit(1)

    if args.all:
        print(f"\nWill test against: {', '.join(available)}")

    tp, tp_name = make_main_player(args.model, local_dir=getattr(args, "local_dir", None))

    # Run matchups
    all_results = []
    for name in available:
        print(f"\n{'='*50}")
        color_note = " [model always white]" if white_only else (" [model always black]" if black_only else "")
        print(f"Matchup: {tp_name} vs {name} ({args.games} games){color_note}")
        print(f"{'='*50}")
        opponent = make_opponent(name, stockfish_path=getattr(args, "stockfish_path", None))
        result = run_matchup(
            tp,
            opponent,
            args.games,
            args.max_half_moves,
            white_only=white_only,
            black_only=black_only,
            tp_name=tp_name,
        )
        print_matchup_result(result, tp_name=tp_name)
        if result["fallbacks_tp"] > 0:
            print(
                f"\n  *** WARNING: {result['fallbacks_tp']} fallback(s) detected! "
                f"Fallbacks hurt your tiebreaker ranking. ***")
        all_results.append(result)
        if hasattr(opponent, "close"):
            opponent.close()  # type: ignore[union-attr]

    if len(all_results) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
