#!/usr/bin/env python3
"""Run 200 games: TransformerPlayer (from HF) vs RandomPlayer."""

import os, sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from collections import Counter
from chess_tournament import Game, RandomPlayer
from player import TransformerPlayer

NUM_GAMES = 200
MAX_HALF_MOVES = 200


def main():
    print("Loading model from HuggingFace: Izzent/recurrent-transformer-chess\n")
    tp = TransformerPlayer("RecurrentTransformer")
    rp = RandomPlayer("Random")

    results = Counter()
    tp_wins = 0
    rp_wins = 0
    draws = 0
    total_fallbacks = 0

    print(f"Running {NUM_GAMES} games (alternating colors), max {MAX_HALF_MOVES} half-moves.\n")

    for i in range(NUM_GAMES):
        white = tp if i % 2 == 0 else rp
        black = rp if i % 2 == 0 else tp
        game = Game(white, black, max_half_moves=MAX_HALF_MOVES)
        outcome, scores, fallbacks = game.play(verbose=False)

        results[outcome] += 1
        total_fallbacks += fallbacks.get("RecurrentTransformer", 0)

        if outcome == "1-0":
            tp_wins += 1 if i % 2 == 0 else 0
            rp_wins += 1 if i % 2 != 0 else 0
        elif outcome == "0-1":
            rp_wins += 1 if i % 2 == 0 else 0
            tp_wins += 1 if i % 2 != 0 else 0
        else:
            draws += 1

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{NUM_GAMES} games done")

    print(f"\nResults over {NUM_GAMES} games vs Random:")
    for key in ["1-0", "0-1", "1/2-1/2", "*"]:
        count = results.get(key, 0)
        if count > 0:
            print(f"  {key}: {count} ({100 * count / NUM_GAMES:.0f}%)")
    print(f"\n  Transformer wins: {tp_wins}")
    print(f"  Random wins:      {rp_wins}")
    print(f"  Draws:            {draws}")
    print(f"  Total fallbacks:  {total_fallbacks}")
    if tp_wins + rp_wins > 0:
        print(f"  Win rate (excl. draws): {100 * tp_wins / (tp_wins + rp_wins):.1f}%")


if __name__ == "__main__":
    main()
