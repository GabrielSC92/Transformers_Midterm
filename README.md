# Transformers Midterm — Recurrent Transformer Chess Player

MSc Applied Data Science, Utrecht University — INFOMTALC 2025/26

## Assignment

Build a `TransformerPlayer` class that takes a chess board state (FEN string) and produces a move (UCI string). All student players compete in a round-robin tournament against each other and against injected baselines (RandomPlayer, Stockfish variants, Mistral-7B, Kimi). Grading is based on leaderboard placement, with up to 10 bonus points for creative solutions.

**Constraints:**
- Must inherit from `Player` (from the [`chess_exam`](https://github.com/bylinina/chess_exam) package)
- Must run on **free-tier Google Colab** (T4 GPU, ~15 GB VRAM)
- No pre-trained chess transformer models allowed
- Model must be pushed to HuggingFace
- `player.py` must contain `TransformerPlayer` initializable with just a name argument
- Fallback counts (illegal moves defaulting to random) are tracked and affect ranking

## Approach: Recurrent Transformer (Universal Transformer)

Instead of fine-tuning an existing LLM, this project trains a **custom transformer from scratch** using a neuroscience-inspired architecture.

### Core Idea

A single shared transformer block is applied **K times recurrently** (with learned iteration embeddings), rather than stacking K unique layers. This mirrors how biological neural networks achieve deep computation through recurrence rather than depth:

```
FEN (char tokens) → [CLS] + Embedding + Positional Encoding
  → Shared Transformer Block × K iterations (+ iteration embedding at each step)
  → [CLS] pooling → Linear classifier → logits over 4,272 UCI moves
  → Mask illegal moves → best legal move
```

### Neuroscience Motivation

| Neuroscience Concept | Transformer Equivalent |
|---|---|
| Lateral connections (BLT "L", Spoerer et al. 2017) | Self-attention within each layer |
| Recurrence (CORnet-s, Kubilius et al. 2019) | Shared transformer block applied K times |
| Time-cycle disambiguation | Learned iteration embeddings |
| 4 recurrent layers ≈ 100+ feedforward layers | K iterations with 1 shared block ≈ K unique layers |

**CORnet-s** demonstrated that a 4-layer recurrent convolutional network matches 100+ layer feedforward networks on ImageNet while achieving the highest Brain-Score. **BLT networks** showed that lateral and top-down connections (implemented as convolutional filters) improve performance on challenging recognition tasks. This architecture translates those principles to transformers.

### Model Specifications

| Parameter | Value |
|---|---|
| Tokenizer | Character-level, 36 tokens (FEN characters + special tokens) |
| d_model | 256 |
| Attention heads | 8 |
| FFN dimension | 1024 |
| Shared iterations (K) | 6 |
| Output classes | 4,272 (all possible UCI moves incl. promotions) |
| Total parameters | ~1.9M |
| Inference | Single forward pass + illegal move masking → guaranteed legal output |

### Why Classification Over Generation?

Autoregressive generation of 4-5 UCI characters multiplies inference time and can produce malformed strings. Classification over all possible source-target square pairs (+ promotion variants) gives a single forward pass with guaranteed valid UCI format. At inference, illegal moves are masked to `-inf` so the output is always legal — **zero fallbacks**.

## Training

- **Data**: [`angeluriot/chess_games`](https://huggingface.co/datasets/angeluriot/chess_games) — 14M games from high-level players, filtered to ELO ≥ 1500
- **Target**: 750K (FEN, UCI move) pairs extracted by replaying games
- **Loss**: Cross-entropy on move classification
- **Optimizer**: AdamW (lr=3e-4) with cosine annealing
- **Batch size**: 512
- **Epochs**: 8
- **Hardware**: Free-tier Colab T4

## Repository Structure

```
├── model.py           # ChessTokenizer, move vocabulary, RecurrentTransformer
├── player.py          # TransformerPlayer (loads model from HF, plays moves)
├── train.ipynb        # Data pipeline, training, evaluation, HF upload
├── requirements.txt   # Dependencies
└── README.md
```

## Usage

**Install the tournament framework:**
```bash
git clone https://github.com/bylinina/chess_exam.git
cd chess_exam && pip install -e . && cd ..
```

**Play a game:**
```python
from chess_tournament import Game, RandomPlayer
from player import TransformerPlayer

tp = TransformerPlayer("RecurrentTransformer")
rp = RandomPlayer("Random")

game = Game(tp, rp, max_half_moves=200)
outcome, scores, fallbacks = game.play()
print(outcome, fallbacks)
```

## HuggingFace Model

Model weights, config, and tokenizer are hosted at [`Izzent/recurrent-transformer-chess`](https://huggingface.co/Izzent/recurrent-transformer-chess). The `TransformerPlayer` downloads them automatically on first use.
