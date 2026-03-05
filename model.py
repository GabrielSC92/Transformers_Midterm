"""
Recurrent Transformer for Chess Move Prediction

A Universal Transformer-style architecture that applies a single shared
transformer block recurrently with iteration embeddings, inspired by
CORnet-s (Kubilius et al.) and BLT networks (Spoerer et al. 2017).

Self-attention serves as lateral connections, weight-sharing across
iterations implements recurrence, and iteration embeddings distinguish
processing time-steps.

Supports two tokenization modes:
  - ChessTokenizer: character-level FEN encoding (legacy)
  - BoardTokenizer: spatial 8x8 board encoding with structured metadata
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
# Spatial Board Tokenizer  (dict-based, separate arrays per token type)
# ---------------------------------------------------------------------------

# Piece IDs for the 64-square board array (vocab size 14)
EMPTY_LIGHT = 0
EMPTY_DARK = 1
PIECE_TO_ID = {
    "P": 2, "N": 3, "B": 4, "R": 5, "Q": 6, "K": 7,   # white
    "p": 8, "n": 9, "b": 10, "r": 11, "q": 12, "k": 13,  # black
}
NUM_PIECE_IDS = 14   # 0-13

# Sequence layout: turn(1) + castling(4) + board(64) + ep(1) = 70 tokens
BOARD_SEQ_LEN = 70
BOARD_START = 5      # index of first board-square token in the 70-token sequence
BOARD_END = 69       # one past last board-square token


def _is_light_square(sq_index: int) -> bool:
    """True when the square is light-coloured (a1=dark, b1=light, ...)."""
    return (sq_index % 8 + sq_index // 8) % 2 == 1


class BoardTokenizer:
    """Spatial tokenizer returning separate arrays per token type.

    ``encode(fen)`` returns a dict with four keys:

    =========  =====  ==================================
    key        shape  description
    =========  =====  ==================================
    board      (64,)  piece IDs 0-13 in a1..h8 order
    turn       (1,)   0 = black, 1 = white
    castling   (4,)   K, Q, k, q — each 0 (no) or 1 (yes)
    ep         (1,)   0 = none, 1-8 = files a-h
    =========  =====  ==================================
    """

    def encode(self, fen: str) -> dict[str, list[int]]:
        parts = fen.split()
        board_str = parts[0]
        side = parts[1] if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 else "-"
        en_passant = parts[3] if len(parts) > 3 else "-"

        board: list[int] = []
        ranks = board_str.split("/")
        for rank_idx in range(7, -1, -1):          # rank 1 … rank 8
            for ch in ranks[rank_idx]:
                if ch.isdigit():
                    for _ in range(int(ch)):
                        sq = len(board)
                        board.append(EMPTY_LIGHT if _is_light_square(sq) else EMPTY_DARK)
                else:
                    board.append(PIECE_TO_ID.get(ch, EMPTY_LIGHT))

        turn = [1 if side == "w" else 0]

        castle = [
            int("K" in castling),
            int("Q" in castling),
            int("k" in castling),
            int("q" in castling),
        ]

        if en_passant == "-":
            ep = [0]
        else:
            ep = [ord(en_passant[0]) - ord("a") + 1]   # 1=a … 8=h

        return {"board": board, "turn": turn, "castling": castle, "ep": ep}

    def to_dict(self) -> dict:
        return {"type": "board_v2"}

    @classmethod
    def from_dict(cls, d: dict) -> "BoardTokenizer":
        return cls()


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor

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
    """Universal Transformer-style model with spatial board encoding.

    Architecture (CORnet-s / BLT inspired, chess-transformers hybrid):
      - Separate embedding tables for pieces, turn, castling, en passant
      - Learned positional embeddings over the 70-token sequence
      - A single SharedTransformerBlock applied K times (+ iteration embeddings)
      - From / To heads: each board-square token is scored as source or dest

    Input sequence layout (70 tokens):
      [0]      turn            (turn_emb,    vocab 2)
      [1-4]    castling K,Q,k,q (castle_emb, vocab 2, shared)
      [5-68]   board a1..h8    (piece_emb,   vocab 14)
      [69]     en passant      (ep_emb,      vocab 9)
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        d_ff: int = 2048,
        num_iterations: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations

        # --- separate embedding tables ---
        self.piece_emb = nn.Embedding(NUM_PIECE_IDS, d_model)   # 14
        self.turn_emb = nn.Embedding(2, d_model)
        self.castle_emb = nn.Embedding(2, d_model)              # shared K/Q/k/q
        self.ep_emb = nn.Embedding(9, d_model)                  # 0=none, 1-8=a-h

        # learned positional embeddings (70 positions)
        self.pos_emb = nn.Embedding(BOARD_SEQ_LEN, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # recurrent transformer block + iteration embeddings
        self.iter_emb = nn.Embedding(num_iterations, d_model)
        self.block = SharedTransformerBlock(d_model, nhead, d_ff, dropout)

        # final norm
        self.final_norm = nn.LayerNorm(d_model)

        # From / To heads — project each board token to a scalar score
        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
        # embeddings: normal with std = 1/sqrt(d_model)
        emb_std = math.pow(self.d_model, -0.5)
        for emb in (self.piece_emb, self.turn_emb, self.castle_emb,
                     self.ep_emb, self.pos_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=emb_std)

    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict with keys ``board`` (B,64), ``turn`` (B,1),
                   ``castling`` (B,4), ``ep`` (B,1).
        Returns:
            from_logits (B, 64), to_logits (B, 64)
        """
        turn_e = self.turn_emb(batch["turn"])           # (B, 1, d)
        castle_e = self.castle_emb(batch["castling"])   # (B, 4, d)
        board_e = self.piece_emb(batch["board"])        # (B, 64, d)
        ep_e = self.ep_emb(batch["ep"])                 # (B, 1, d)

        x = torch.cat([turn_e, castle_e, board_e, ep_e], dim=1)  # (B, 70, d)
        x = x + self.pos_emb.weight.unsqueeze(0)
        x = x * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        device = x.device
        iter_indices = torch.arange(self.num_iterations, device=device)
        for t in range(self.num_iterations):
            x = x + self.iter_emb(iter_indices[t]).unsqueeze(0).unsqueeze(0)
            x = self.block(x)

        x = self.final_norm(x)

        board_repr = x[:, BOARD_START:BOARD_END, :]           # (B, 64, d)
        from_logits = self.from_head(board_repr).squeeze(-1)  # (B, 64)
        to_logits = self.to_head(board_repr).squeeze(-1)      # (B, 64)

        return from_logits, to_logits

    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "d_model": self.d_model,
            "nhead": self.block.self_attn.num_heads,
            "d_ff": self.block.ff[0].out_features,
            "num_iterations": self.num_iterations,
            "dropout": self.emb_dropout.p,
        }

    @classmethod
    def from_config(cls, config: dict) -> "RecurrentTransformer":
        return cls(
            d_model=config["d_model"],
            nhead=config["nhead"],
            d_ff=config["d_ff"],
            num_iterations=config["num_iterations"],
            dropout=config.get("dropout", 0.1),
        )
