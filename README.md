# Transformers Midterm - Recurrent Transformer Chess Player

MSc Applied Data Science, Utrecht University - INFOMTALC 2025/26

## Assignment

Build a TransformerPlayer class that takes a chess board state (FEN string) and produces a move (UCI string). All student players compete in a round-robin tournament against each other and against baselines (RandomPlayer, Stockfish variants, Mistral-7B, Kimi). Grading is based on leaderboard placement.

Constraints: must inherit from Player (from the chess_exam package), run on free-tier Google Colab (T4), no pre-trained chess transformers, model on HuggingFace, player.py with TransformerPlayer initializable with just a name, and fallback counts affect ranking.

## Approach

Custom transformer trained from scratch. Two main ideas:

- From/To square prediction (from sgrvinod/chess-transformers): separate embeddings for pieces, turn, castling, en passant; two linear heads score each of the 64 squares as source and destination. At inference we score each legal move as from_logits[src] + to_logits[dst] and pick the best.

- Recurrent weight sharing (CORnet-s / BLT, from HNA course): one transformer block applied K times with learned iteration embeddings instead of K stacked layers.

Board is encoded as 70 tokens (turn, castling x4, 64 squares, en passant). Spatial encoding with 14 piece IDs. Model: d_model 512, 8 heads, FFN 2048, 8 iterations, ~3.2M params. Single forward pass, output is always a legal move (zero fallbacks).

## Data

generate_engine_data.py uses local Stockfish to get best moves for positions from angeluriot/chess_games (ELO >= 1500). Stockfish depth 10, 8 workers, output 2M (FEN, UCI) pairs in JSONL.

## Training

Cross-entropy on from-square and to-square. AdamW, Vaswani LR schedule (warmup 4000 steps). Batch 512, 16 epochs with early stopping (patience 2). Trained locally on 5070Ti; runs on Colab T4 for inference.

## Repo

model_2.py (BoardTokenizer, RecurrentTransformer), player.py (TransformerPlayer), train.ipynb (data, training, HF upload), generate_engine_data.py, run_test.py (local tests vs Random/Stockfish etc), requirements.txt.

## Usage

Clone chess_exam, pip install -e . in it. Then:

```python
from chess_tournament import Game, RandomPlayer
from player import TransformerPlayer

tp = TransformerPlayer("RecurrentTransformer")
rp = RandomPlayer("Random")
game = Game(tp, rp, max_half_moves=200)
outcome, scores, fallbacks = game.play()
print(outcome, fallbacks)
```

Model is on HuggingFace at Izzent/recurrent-transformer-chess. Player downloads it on first use.
