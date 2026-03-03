"""
LM-based chess player: loads a LoRA fine-tuned causal LM (from train_lm.py),
prompts with FEN, generates move as text, parses UCI and validates legal. No random fallback.
Use for matchup vs TransformerPlayer (encoder-only).
"""

import re
from pathlib import Path
from typing import Optional

import chess
import torch
from chess_tournament import Player
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


class TransformerPlayerLM(Player):
    """Player using a fine-tuned causal LM (LoRA) from train_lm.py. Load from local dir or HF."""

    def __init__(
        self,
        name: str = "TransformerLM",
        model_dir: str = "chess_lm_lora",
        base_model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: Optional[str] = None,
    ):
        super().__init__(name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir if self.model_dir.exists() else model_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, str(self.model_dir) if self.model_dir.exists() else model_dir)
        self.model.eval()

    @torch.no_grad()
    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        prompt = f"FEN: {fen}\nMove:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt) :].strip()

        match = UCI_RE.search(decoded)
        if not match:
            return None
        move_str = match.group(0)
        try:
            move = chess.Move.from_uci(move_str)
        except ValueError:
            return None
        if move in board.legal_moves:
            return move_str
        return None
