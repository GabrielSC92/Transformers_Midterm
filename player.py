"""TransformerPlayer for the INFOMTALC chess tournament."""

import json
from pathlib import Path
from typing import Optional

import chess
import torch
from huggingface_hub import hf_hub_download

from chess_tournament import Player
from model_2 import BoardTokenizer, RecurrentTransformer

HF_REPO = "Izzent/recurrent-transformer-chess"


class TransformerPlayer(Player):

    def __init__(
        self,
        name: str = "RecurrentTransformer",
        repo_id: str = HF_REPO,
        device=None,
        local_dir=None,
    ):
        super().__init__(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # try loading from local directory first, fall back to HF Hub
        config_path = None
        weights_path = None

        if local_dir is not None:
            base = Path(local_dir)
            wp = base / "model.pt"
            if not wp.exists():
                wp = base / "best_model.pt"
            if (base / "config.json").exists() and wp.exists():
                config_path = str(base / "config.json")
                weights_path = str(wp)

        if config_path is None:
            config_path = hf_hub_download(repo_id, "config.json")
            weights_path = hf_hub_download(repo_id, "model.pt")

        with open(config_path) as f:
            config = json.load(f)

        self.tokenizer = BoardTokenizer()
        self.model = RecurrentTransformer.from_config(config)
        assert weights_path is not None
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        enc = self.tokenizer.encode(fen)
        batch = {
            "board": torch.tensor([enc["board"]], dtype=torch.long, device=self.device),
            "turn": torch.tensor([enc["turn"]], dtype=torch.long, device=self.device),
            "castling": torch.tensor([enc["castling"]], dtype=torch.long, device=self.device),
            "ep": torch.tensor([enc["ep"]], dtype=torch.long, device=self.device),
        }

        with torch.no_grad():
            from_logits, to_logits = self.model(batch)
        from_logits = from_logits.squeeze(0)
        to_logits = to_logits.squeeze(0)

        best_score = float("-inf")
        best_move = None
        for move in legal_moves:
            score = from_logits[move.from_square].item() + to_logits[move.to_square].item()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci() if best_move else None
