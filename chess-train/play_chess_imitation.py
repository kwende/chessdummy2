#!/usr/bin/env python3
"""
play_chess_imitation.py

Interactive CLI for testing a trained chess imitation model.

Features:
- Play against the model from the terminal
- Human can enter SAN ("e4", "Nf3", "Qxe5+") or UCI ("e2e4")
- Shows top-k model move candidates with probabilities
- Supports commands: help, board, fen, top, undo, quit

Usage examples:
  python3 play_chess_imitation.py --ckpt ckpts/latest.pt
  python3 play_chess_imitation.py --ckpt ckpts/latest.pt --color black
  python3 play_chess_imitation.py --ckpt ckpts/latest.pt --temperature 0.8 --topk 5
"""

import argparse
import math
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
import chess

from chess_imitation import (
    PolicyNet,
    board_to_tensor,
    legal_mask,
    decode_move,
)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[PolicyNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    channels = int(saved_args.get("channels", 128))
    blocks = int(saved_args.get("blocks", 6))

    model = PolicyNet(in_planes=19, channels=channels, blocks=blocks).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, saved_args


@torch.no_grad()
def get_masked_logits(model: PolicyNet, board: chess.Board, mover_rating: float, device: torch.device) -> torch.Tensor:
    x = board_to_tensor(board, mover_rating).unsqueeze(0).to(device)
    logits = model(x)[0]
    lm = legal_mask(board).to(device)
    logits = logits.masked_fill(~lm, -1e9)
    return logits


@torch.no_grad()
def top_moves(
    model: PolicyNet,
    board: chess.Board,
    mover_rating: float,
    device: torch.device,
    topk: int = 5,
    temperature: float = 1.0,
) -> List[Tuple[chess.Move, float]]:
    logits = get_masked_logits(model, board, mover_rating, device)

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=0)
    else:
        probs = F.softmax(logits, dim=0)

    values, indices = torch.topk(probs, k=min(topk, probs.numel()))
    out = []
    for p, idx in zip(values.tolist(), indices.tolist()):
        mv = decode_move(idx)
        if mv in board.legal_moves:
            out.append((mv, p))
    return out


@torch.no_grad()
def sample_model_move(
    model: PolicyNet,
    board: chess.Board,
    mover_rating: float,
    device: torch.device,
    temperature: float = 1.0,
) -> chess.Move:
    logits = get_masked_logits(model, board, mover_rating, device)

    if temperature <= 0:
        idx = int(torch.argmax(logits).item())
        mv = decode_move(idx)
        if mv in board.legal_moves:
            return mv
        return next(iter(board.legal_moves))

    probs = F.softmax(logits / temperature, dim=0)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    mv = decode_move(idx)

    if mv in board.legal_moves:
        return mv

    # fallback
    idx = int(torch.argmax(logits).item())
    mv = decode_move(idx)
    if mv in board.legal_moves:
        return mv

    return next(iter(board.legal_moves))


def print_board(board: chess.Board) -> None:
    print()
    print(board)
    print()
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print()


def parse_user_move(board: chess.Board, text: str) -> chess.Move:
    text = text.strip()

    # Try SAN first, since that's what humans usually type.
    try:
        return board.parse_san(text)
    except Exception:
        pass

    # Then try UCI.
    try:
        mv = chess.Move.from_uci(text)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    raise ValueError(f"Could not parse move: {text}")


def move_to_pretty(board: chess.Board, mv: chess.Move) -> str:
    try:
        return board.san(mv)
    except Exception:
        return mv.uci()


def print_topk(model, board, mover_rating, device, topk, temperature) -> None:
    candidates = top_moves(
        model=model,
        board=board,
        mover_rating=mover_rating,
        device=device,
        topk=topk,
        temperature=temperature,
    )
    print(f"Top {len(candidates)} model candidates:")
    for i, (mv, p) in enumerate(candidates, 1):
        pretty = move_to_pretty(board, mv)
        print(f"  {i:>2}. {pretty:<12} {mv.uci():<8} p={p:.4f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. ckpts/latest.pt")
    ap.add_argument("--color", choices=["white", "black"], default="white", help="Human side")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for model")
    ap.add_argument("--topk", type=int, default=5, help="How many candidate moves to show")
    ap.add_argument("--model-rating", type=float, default=None,
                    help="Mover rating fed into the model. Defaults to midpoint of training band if available.")
    ap.add_argument("--start-fen", default="", help="Optional starting FEN")
    ap.add_argument("--device", default="", help="cuda, cpu, cuda:0, etc. Default: auto")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, saved_args = load_model(args.ckpt, device)

    min_rating = saved_args.get("min_rating")
    max_rating = saved_args.get("max_rating")
    if args.model_rating is not None:
        mover_rating = float(args.model_rating)
    elif min_rating is not None and max_rating is not None:
        mover_rating = 0.5 * (float(min_rating) + float(max_rating))
    else:
        mover_rating = 1000.0

    board = chess.Board(args.start_fen) if args.start_fen else chess.Board()
    human_color = chess.WHITE if args.color == "white" else chess.BLACK

    print()
    print("Loaded model.")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Model rating input: {mover_rating}")
    print(f"Human plays: {args.color}")
    if min_rating is not None and max_rating is not None:
        print(f"Training band from checkpoint args: {min_rating}-{max_rating}")
    print()
    print("Commands: help, board, fen, top, undo, quit")
    print("Moves: SAN like e4 / Nf3 / Qxe5+, or UCI like e2e4")
    print()

    history = []

    while not board.is_game_over():
        print_board(board)

        if board.turn == human_color:
            while True:
                text = input("Your move > ").strip()

                if not text:
                    continue

                lower = text.lower()

                if lower in ("quit", "exit", "q"):
                    print("Exiting.")
                    return

                if lower in ("help", "?"):
                    print("Commands: help, board, fen, top, undo, quit")
                    print("Moves may be SAN (e4, Nf3, Qxe5+) or UCI (e2e4).")
                    print()
                    continue

                if lower == "board":
                    print_board(board)
                    continue

                if lower == "fen":
                    print(board.fen())
                    print()
                    continue

                if lower == "top":
                    print_topk(model, board, mover_rating, device, args.topk, args.temperature)
                    continue

                if lower == "undo":
                    if len(history) >= 1:
                        board.pop()
                        history.pop()
                    if len(history) >= 1 and board.turn != human_color:
                        board.pop()
                        history.pop()
                    print("Undid move(s).")
                    continue

                try:
                    mv = parse_user_move(board, text)
                    san = move_to_pretty(board, mv)
                    history.append(mv)
                    board.push(mv)
                    print(f"You played: {san} ({mv.uci()})")
                    print()
                    break
                except Exception as e:
                    print(f"Invalid move: {e}")
                    print()
        else:
            print_topk(model, board, mover_rating, device, args.topk, args.temperature)
            mv = sample_model_move(
                model=model,
                board=board,
                mover_rating=mover_rating,
                device=device,
                temperature=args.temperature,
            )
            san = move_to_pretty(board, mv)
            history.append(mv)
            board.push(mv)
            print(f"Model plays: {san} ({mv.uci()})")
            print()

    print_board(board)
    print("Game over.")
    print(f"Result: {board.result()}")
    print(f"Outcome: {board.outcome()}")


if __name__ == "__main__":
    main()