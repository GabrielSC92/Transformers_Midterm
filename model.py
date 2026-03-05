# Recurrent Transformer for chess move prediction
# got the from/to square idea from sgrvinod/chess-transformers,
# but uses a shared block applied multiple times (like CORnet-s does
# for vision — same weights reused across iterations)

import math
import torch
import torch.nn as nn


# all possible UCI moves (from-square + to-square + optional promotion)
SQUARE_NAMES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
PROMOTION_PIECES = ["q", "r", "b", "n"]


def build_move_vocabulary():
    moves = []
    move_to_idx = {}

    # regular moves: every from-to pair
    for from_sq in range(64):
        for to_sq in range(64):
            uci = SQUARE_NAMES[from_sq] + SQUARE_NAMES[to_sq]
            move_to_idx[uci] = len(moves)
            moves.append(uci)

    # promotions: pawn on 7th rank moving to 8th (white) or 2nd to 1st (black)
    for from_sq in range(64):
        for to_sq in range(64):
            to_rank = to_sq // 8
            from_rank = from_sq // 8
            from_file = from_sq % 8
            to_file = to_sq % 8
            if to_rank not in (0, 7):
                continue
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


# board encoding (based on sgrvinod/chess-transformers)
# 14 piece IDs, sequence = turn(1) + castling(4) + board(64) + ep(1) = 70 tokens

PIECE_TO_ID = {
    "P": 2, "N": 3, "B": 4, "R": 5, "Q": 6, "K": 7,
    "p": 8, "n": 9, "b": 10, "r": 11, "q": 12, "k": 13,
}
NUM_PIECE_IDS = 14
BOARD_SEQ_LEN = 70
BOARD_START = 5   # first board-square token index
BOARD_END = 69    # one past last board-square token


def _is_light_square(sq_index):
    return (sq_index % 8 + sq_index // 8) % 2 == 1


class BoardTokenizer:
    """Encodes a FEN string into separate arrays for board, turn, castling, ep."""

    def encode(self, fen):
        parts = fen.split()
        board_str = parts[0]
        side = parts[1] if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 else "-"
        en_passant = parts[3] if len(parts) > 3 else "-"

        board = []
        ranks = board_str.split("/")
        for rank_idx in range(7, -1, -1):
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
            ep = [ord(en_passant[0]) - ord("a") + 1]

        return {"board": board, "turn": turn, "castling": castle, "ep": ep}

    def to_dict(self):
        return {"type": "board_v2"}

    @classmethod
    def from_dict(cls, d):
        return cls()


EMPTY_LIGHT = 0
EMPTY_DARK = 1


# single transformer block, reused across all iterations
class SharedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
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

    def forward(self, x, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


# the actual model — like sgrvinod's ChessTransformerEncoderFT but with
# shared weights across iterations instead of separate layers
# (idea from CORnet-s: 4 recurrent layers ~ 100+ feedforward layers)
class RecurrentTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, d_ff=2048, num_iterations=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations

        # embeddings for each part of the board state
        self.piece_emb = nn.Embedding(NUM_PIECE_IDS, d_model)
        self.turn_emb = nn.Embedding(2, d_model)
        self.castle_emb = nn.Embedding(2, d_model)
        self.ep_emb = nn.Embedding(9, d_model)

        self.pos_emb = nn.Embedding(BOARD_SEQ_LEN, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # one block reused K times, with a different learned embedding each time
        # so the model knows which iteration it's on
        self.iter_emb = nn.Embedding(num_iterations, d_model)
        self.block = SharedTransformerBlock(d_model, nhead, d_ff, dropout)

        self.final_norm = nn.LayerNorm(d_model)

        # predict from-square and to-square separately
        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
        emb_std = math.pow(self.d_model, -0.5)
        for emb in (self.piece_emb, self.turn_emb, self.castle_emb,
                     self.ep_emb, self.pos_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=emb_std)

    def forward(self, batch):
        turn_e = self.turn_emb(batch["turn"])           # (B, 1, d)
        castle_e = self.castle_emb(batch["castling"])   # (B, 4, d)
        board_e = self.piece_emb(batch["board"])        # (B, 64, d)
        ep_e = self.ep_emb(batch["ep"])                 # (B, 1, d)

        # 70 tokens: turn + 4 castling + 64 board squares + en passant
        x = torch.cat([turn_e, castle_e, board_e, ep_e], dim=1)
        x = x + self.pos_emb.weight.unsqueeze(0)
        x = x * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        # run the same block multiple times (like CORnet-s recurrence)
        iter_indices = torch.arange(self.num_iterations, device=x.device)
        for t in range(self.num_iterations):
            x = x + self.iter_emb(iter_indices[t]).unsqueeze(0).unsqueeze(0)
            x = self.block(x)

        x = self.final_norm(x)

        # grab just the 64 board-square tokens
        board_repr = x[:, BOARD_START:BOARD_END, :]
        from_logits = self.from_head(board_repr).squeeze(-1)  # (B, 64)
        to_logits = self.to_head(board_repr).squeeze(-1)      # (B, 64)

        return from_logits, to_logits

    def get_config(self):
        return {
            "d_model": self.d_model,
            "nhead": self.block.self_attn.num_heads,
            "d_ff": self.block.ff[0].out_features,
            "num_iterations": self.num_iterations,
            "dropout": self.emb_dropout.p,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            d_model=config["d_model"],
            nhead=config["nhead"],
            d_ff=config["d_ff"],
            num_iterations=config["num_iterations"],
            dropout=config.get("dropout", 0.1),
        )
