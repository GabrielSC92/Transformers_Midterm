"""TransformerPlayer for the INFOMTALC chess tournament.

Uses a Hybrid Spatial Recurrent Transformer with From/To prediction heads.
Each legal move is scored as  from_logits[from_sq] + to_logits[to_sq]  and
the highest-scoring legal move is returned. Promotions default to queen.
"""

import json
import os
from pathlib import Path
from typing import Optional

import chess
import torch
from huggingface_hub import hf_hub_download

from chess_tournament import Player
from model import BoardTokenizer, RecurrentTransformer

HF_REPO = "Izzent/recurrent-transformer-chess"


class TransformerPlayer(Player):

    def __init__(
        self,
        name: str = "RecurrentTransformer",
        repo_id: str = HF_REPO,
        device: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        super().__init__(name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if local_dir is not None:
            base = Path(local_dir).resolve()
            config_path = base / "config.json"
            tokenizer_path = base / "tokenizer.json"
            weights_path = base / "model.pt"
            if not weights_path.exists():
                weights_path = base / "best_model.pt"
            if config_path.exists() and tokenizer_path.exists(
            ) and weights_path.exists():
                config_path = str(config_path)
                tokenizer_path = str(tokenizer_path)
                weights_path = str(weights_path)
            else:
                local_dir = None
        if local_dir is None:
            config_path = hf_hub_download(repo_id, "config.json")
            weights_path = hf_hub_download(repo_id, "model.pt")
            tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")

        with open(config_path) as f:
            config = json.load(f)

        self.tokenizer = BoardTokenizer()
        self.model = RecurrentTransformer.from_config(config)
        state_dict = torch.load(weights_path,
                                map_location=self.device,
                                weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _encode_fen(self, fen: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer.encode(fen)
        return {
            "board": torch.tensor([enc["board"]], dtype=torch.long, device=self.device),
            "turn": torch.tensor([enc["turn"]], dtype=torch.long, device=self.device),
            "castling": torch.tensor([enc["castling"]], dtype=torch.long, device=self.device),
            "ep": torch.tensor([enc["ep"]], dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        batch = self._encode_fen(fen)
        from_logits, to_logits = self.model(batch)
        from_logits = from_logits.squeeze(0)   # (64,)
        to_logits = to_logits.squeeze(0)       # (64,)

        best_score = float("-inf")
        best_move = None
        for move in legal_moves:
            score = from_logits[move.from_square].item() + to_logits[move.to_square].item()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci() if best_move is not None else None
