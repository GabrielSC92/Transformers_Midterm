# generate training data: for each FEN, get stockfish's best move

import chess
import chess.engine
import json
from datasets import load_dataset
from multiprocessing import Pool
from tqdm import tqdm

# config
OUTPUT_FILE = "engine_data_test.jsonl"
TARGET_POSITIONS = 10000
MIN_ELO = 0
STOCKFISH_PATH = r"C:\Users\Gabri\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
DEPTH = 10
THREADS = 3
HASH_MB = 128
NUM_WORKERS = 8


# each worker opens its own stockfish process
stockfish = None

def start_stockfish():
    global stockfish
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    stockfish.configure({"Threads": THREADS, "Hash": HASH_MB})

def get_best_move(fen):
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    result = stockfish.play(board, chess.engine.Limit(depth=DEPTH))  # type: ignore
    if result.move is None:
        return None
    return (fen, result.move.uci())


def collect_fens():
    """Collect FENs by replaying games from the HuggingFace dataset."""
    fens = []
    dataset = load_dataset("angeluriot/chess_games", split="train", streaming=True)

    for game in dataset:
        if len(fens) >= TARGET_POSITIONS:
            break

        white_elo = int(game.get("white_elo") or 0)
        black_elo = int(game.get("black_elo") or 0)
        if white_elo < MIN_ELO or black_elo < MIN_ELO:
            continue

        moves = game.get("moves_uci")
        if not moves:
            continue

        board = chess.Board()
        for move_uci in moves:
            if len(fens) >= TARGET_POSITIONS or board.is_game_over():
                break
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    break
            except Exception:
                break
            fens.append(board.fen())
            board.push(move)

    return fens


if __name__ == "__main__":
    print("Collecting FENs from dataset...")
    fens = collect_fens()
    print(f"Collected {len(fens)} positions, starting Stockfish analysis...")

    pool = Pool(NUM_WORKERS, initializer=start_stockfish)
    results = pool.imap_unordered(get_best_move, fens, chunksize=256)

    count = 0
    f = open(OUTPUT_FILE, "w")
    for pair in tqdm(results, total=len(fens)):
        if pair is not None:
            fen, uci = pair
            f.write(json.dumps({"fen": fen, "uci": uci}) + "\n")
            count += 1
    f.close()
    pool.close()

    print(f"Done. Wrote {count} positions to {OUTPUT_FILE}")
