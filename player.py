"""TransformerPlayer for the INFOMTALC chess tournament.

Uses a custom RecurrentTransformer (Universal Transformer-style with shared
weights and iteration embeddings) trained from scratch on FEN → UCI move
classification.  At inference, illegal moves are masked to -inf so the
output is always a legal move — zero fallbacks.
"""

import json
import os
from typing import Optional

import chess
import torch
from huggingface_hub import hf_hub_download

from chess_tournament import Player
from model import (
    ChessTokenizer,
    RecurrentTransformer,
    MOVE_VOCAB,
    MOVE_TO_IDX,
)

HF_REPO = "Izzent/recurrent-transformer-chess"


class TransformerPlayer(Player):
    def __init__(
        self,
        name: str = "RecurrentTransformer",
        repo_id: str = HF_REPO,
        device: Optional[str] = None,
        lookahead_plies: int = 0,
        lookahead_top_k: int = 5,
    ):
        super().__init__(name)
        self.lookahead_plies = lookahead_plies
        self.lookahead_top_k = lookahead_top_k
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        config_path = hf_hub_download(repo_id, "config.json")
        weights_path = hf_hub_download(repo_id, "model.pt")
        tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")

        with open(config_path) as f:
            config = json.load(f)
        with open(tokenizer_path) as f:
            tok_dict = json.load(f)

        self.tokenizer = ChessTokenizer.from_dict(tok_dict)
        self.model = RecurrentTransformer.from_config(config)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _score_position(self, fen: str) -> float:
        """Return model's max logit over legal moves for the side to move (higher = they have a strong move)."""
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return float("-inf")
        tokens = self.tokenizer.encode(fen)
        input_ids = torch.tensor([tokens], device=self.device)
        logits = self.model(input_ids).squeeze(0)
        legal_uci = {m.uci() for m in legal_moves}
        best = float("-inf")
        for uci in legal_uci:
            idx = MOVE_TO_IDX.get(uci)
            if idx is not None and logits[idx].item() > best:
                best = logits[idx].item()
        return best

    @torch.no_grad()
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        tokens = self.tokenizer.encode(fen)
        input_ids = torch.tensor([tokens], device=self.device)
        logits = self.model(input_ids).squeeze(0)  # (num_moves,)

        # Mask illegal moves to -inf
        legal_uci = {m.uci() for m in legal_moves}
        mask = torch.full_like(logits, float("-inf"))
        for uci in legal_uci:
            idx = MOVE_TO_IDX.get(uci)
            if idx is not None:
                mask[idx] = 0.0
        logits = logits + mask

        if self.lookahead_plies <= 0:
            best_idx = logits.argmax().item()
            return MOVE_VOCAB[best_idx]

        # 1-ply lookahead: score each of our top-k moves by the resulting position (opponent to move).
        # We want to minimize opponent's best response, so score = -opponent_max_logit.
        top_k = min(self.lookahead_top_k, len(legal_moves))
        move_scores = []
        for m in legal_moves:
            idx = MOVE_TO_IDX.get(m.uci())
            if idx is None:
                continue
            move_scores.append((m, logits[idx].item()))
        move_scores.sort(key=lambda x: -x[1])
        move_scores = move_scores[:top_k]

        best_move = move_scores[0][0]
        best_score = float("-inf")
        for m, _ in move_scores:
            board.push(m)
            opp_score = self._score_position(board.fen())
            board.pop()
            # We want positions where opponent is worse: minimize their max logit
            our_score = -opp_score
            if our_score > best_score:
                best_score = our_score
                best_move = m
        return best_move.uci()
