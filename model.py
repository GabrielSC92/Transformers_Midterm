"""
Recurrent Transformer for Chess Move Prediction

A Universal Transformer-style architecture that applies a single shared
transformer block recurrently with iteration embeddings, inspired by
CORnet-s (Kubilius et al.) and BLT networks (Spoerer et al. 2017).

Self-attention serves as lateral connections, weight-sharing across
iterations implements recurrence, and iteration embeddings distinguish
processing time-steps.
"""

import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Move vocabulary
# ---------------------------------------------------------------------------

SQUARE_NAMES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
# a1=0, b1=1 ... h1=7, a2=8 ... h8=63  (python-chess order)

PROMOTION_PIECES = ["q", "r", "b", "n"]


def build_move_vocabulary():
    """Build a deterministic list of all possible UCI move strings.

    Regular moves: 4096 (from_sq x to_sq including impossible pairs).
    Promotions:    176  (pawn moves to rank 1 or 8 with q/r/b/n suffix).
    """
    moves = []
    move_to_idx = {}

    for from_sq in range(64):
        for to_sq in range(64):
            uci = SQUARE_NAMES[from_sq] + SQUARE_NAMES[to_sq]
            move_to_idx[uci] = len(moves)
            moves.append(uci)

    for from_sq in range(64):
        for to_sq in range(64):
            to_rank = to_sq // 8  # 0-indexed rank (0 = rank 1, 7 = rank 8)
            from_rank = from_sq // 8
            from_file = from_sq % 8
            to_file = to_sq % 8
            if to_rank not in (0, 7):
                continue
            # White promotion: from rank 7 (idx 6) to rank 8 (idx 7)
            # Black promotion: from rank 2 (idx 1) to rank 1 (idx 0)
            if not ((from_rank == 6 and to_rank == 7) or
                    (from_rank == 1 and to_rank == 0)):
                continue
            if abs(from_file - to_file) > 1:
                continue
            for piece in PROMOTION_PIECES:
                uci = SQUARE_NAMES[from_sq] + SQUARE_NAMES[to_sq] + piece
                move_to_idx[uci] = len(moves)
                moves.append(uci)

    return moves, move_to_idx


MOVE_VOCAB, MOVE_TO_IDX = build_move_vocabulary()
NUM_MOVES = len(MOVE_VOCAB)


# ---------------------------------------------------------------------------
# Character-level FEN tokenizer
# ---------------------------------------------------------------------------

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"

FEN_CHARS = sorted(set(
    list("abcdefghkKnNpPqQrRwB") +     # pieces + colors + bishop
    list("0123456789") +                  # digits (ranks, counters)
    ["/", "-", " "]                       # separators
))


class ChessTokenizer:
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.special_tokens = [PAD_TOKEN, CLS_TOKEN, UNK_TOKEN]
        self.vocab = self.special_tokens + FEN_CHARS
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.cls_id = self.token_to_id[CLS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.vocab_size = len(self.vocab)

    def encode(self, fen: str) -> list[int]:
        tokens = [self.cls_id]
        for ch in fen:
            tokens.append(self.token_to_id.get(ch, self.unk_id))
        tokens = tokens[: self.max_length]
        while len(tokens) < self.max_length:
            tokens.append(self.pad_id)
        return tokens

    def to_dict(self) -> dict:
        return {"max_length": self.max_length, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "ChessTokenizer":
        tok = cls(max_length=d["max_length"])
        tok.vocab = d["vocab"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.id_to_token = {i: t for i, t in enumerate(tok.vocab)}
        tok.pad_id = tok.token_to_id[PAD_TOKEN]
        tok.cls_id = tok.token_to_id[CLS_TOKEN]
        tok.unk_id = tok.token_to_id[UNK_TOKEN]
        tok.vocab_size = len(tok.vocab)
        return tok


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SharedTransformerBlock(nn.Module):
    """Single transformer encoder block reused across iterations."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)
        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class RecurrentTransformer(nn.Module):
    """Universal Transformer-style model with shared weights across iterations.

    Architecture (CORnet-s / BLT inspired):
      - Character-level token embeddings + sinusoidal positional encoding
      - A single SharedTransformerBlock applied K times
      - Learned iteration embeddings added at each step
      - [CLS] token pooling → classification head over all UCI moves
    """

    def __init__(
        self,
        vocab_size: int = 36,
        d_model: int = 256,
        nhead: int = 8,
        d_ff: int = 1024,
        num_iterations: int = 6,
        num_moves: int = NUM_MOVES,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.num_iterations = num_iterations
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)

        # Learned iteration embeddings (broadcast across sequence positions)
        self.iter_emb = nn.Embedding(num_iterations, d_model)

        # Single shared block
        self.block = SharedTransformerBlock(d_model, nhead, d_ff, dropout)

        # Final layer norm + classification head
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_moves)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token ids with [CLS] at position 0
        Returns:
            logits: (batch, num_moves)
        """
        pad_mask = input_ids == self.pad_token_id  # True = ignore

        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.emb_dropout(x)

        iter_indices = torch.arange(
            self.num_iterations, device=input_ids.device
        )
        for t in range(self.num_iterations):
            # Add iteration embedding (broadcast over batch & seq)
            x = x + self.iter_emb(iter_indices[t]).unsqueeze(0).unsqueeze(0)
            x = self.block(x, key_padding_mask=pad_mask)

        x = self.final_norm(x)
        cls_repr = x[:, 0]  # [CLS] position
        logits = self.classifier(cls_repr)
        return logits

    def get_config(self) -> dict:
        return {
            "vocab_size": self.token_emb.num_embeddings,
            "d_model": self.d_model,
            "nhead": self.block.self_attn.num_heads,
            "d_ff": self.block.ff[0].out_features,
            "num_iterations": self.num_iterations,
            "num_moves": self.classifier.out_features,
            "max_seq_len": self.pos_enc.pe.size(1),
            "pad_token_id": self.pad_token_id,
        }

    @classmethod
    def from_config(cls, config: dict) -> "RecurrentTransformer":
        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            d_ff=config["d_ff"],
            num_iterations=config["num_iterations"],
            num_moves=config["num_moves"],
            max_seq_len=config["max_seq_len"],
            pad_token_id=config["pad_token_id"],
        )
