#!/usr/bin/env python3
"""
Run games between encoder-only TransformerPlayer (player.py) and LM-based TransformerPlayerLM (player_lm.py).
Use this to compare which approach shows more promise before focusing on one for submission.

Requires: chess_exam installed (pip install -e . in chess_exam repo). Run from this repo root.

  python run_matchup_encoder_vs_lm.py --games 20 --max-moves 200
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chess_tournament import Game

from player import TransformerPlayer
from player_lm import TransformerPlayerLM


def main():
    p = argparse.ArgumentParser(description="Encoder (RecurrentTransformer) vs LM (fine-tuned causal LM)")
    p.add_argument("--games", type=int, default=20, help="Number of games (each color)")
    p.add_argument("--max-moves", type=int, default=200, help="Max half-moves per game")
    p.add_argument("--lm-dir", default="chess_lm_lora", help="Dir for LoRA adapter (from train_lm.py)")
    args = p.parse_args()

    encoder = TransformerPlayer("Encoder")
    lm = TransformerPlayerLM("LM", model_dir=args.lm_dir)

    encoder_wins = 0
    lm_wins = 0
    draws = 0
    fallbacks_encoder = 0
    fallbacks_lm = 0

    print(f"Running {args.games * 2} games (alternating colors), max {args.max_moves} half-moves.\n")

    for i in range(args.games * 2):
        white = encoder if i % 2 == 0 else lm
        black = lm if i % 2 == 0 else encoder
        game = Game(white, black, max_half_moves=args.max_moves)
        outcome, scores, fallbacks = game.play(verbose=False)

        fallbacks_encoder += fallbacks.get("Encoder", 0)
        fallbacks_lm += fallbacks.get("LM", 0)

        if outcome == "1-0":
            if i % 2 == 0:
                encoder_wins += 1
                winner = "Encoder"
            else:
                lm_wins += 1
                winner = "LM"
        elif outcome == "0-1":
            if i % 2 == 0:
                lm_wins += 1
                winner = "LM"
            else:
                encoder_wins += 1
                winner = "Encoder"
        else:
            draws += 1
            winner = "Draw"

        print(f"  Game {i + 1:2d}: {outcome:8s}  Winner: {winner:8s}  Fallbacks Encoder={fallbacks.get('Encoder', 0)} LM={fallbacks.get('LM', 0)}")

    print("\n" + "=" * 60)
    print("SUMMARY (Encoder vs LM)")
    print("=" * 60)
    print(f"  Encoder (RecurrentTransformer) wins: {encoder_wins}")
    print(f"  LM (fine-tuned causal LM) wins:      {lm_wins}")
    print(f"  Draws:                                {draws}")
    print(f"  Total fallbacks Encoder:              {fallbacks_encoder}")
    print(f"  Total fallbacks LM:                   {fallbacks_lm}")
    print("=" * 60)
    total_decided = encoder_wins + lm_wins
    if total_decided > 0:
        print(f"  Encoder win rate (excluding draws): {100 * encoder_wins / total_decided:.1f}%")
        print(f"  LM win rate (excluding draws):     {100 * lm_wins / total_decided:.1f}%")


if __name__ == "__main__":
    main()
