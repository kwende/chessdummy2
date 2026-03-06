#!/usr/bin/env python3
"""
chess_imitation.py

Train a human-move imitation policy network from Lichess PGN.zst files, filtered by rating band.
- Streams .pgn.zst on the fly (no full decompression required)
- CNN/ResNet policy network
- Illegal-move masking in both training + inference
- TensorBoard logging (loss, acc, entropy, lr, throughput)
- Checkpointing with clean resume (model+optimizer+scaler+step), atomic latest.pt saves
- Graceful save on SIGINT/SIGTERM (e.g., desktop interruption)

Dependencies:
  pip install python-chess zstandard numpy tqdm tensorboard
  + PyTorch with CUDA (for 2080 Ti)
"""

import argparse
import io
import os
import random
import signal
import tempfile
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd
import chess
import chess.pgn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ----------------------------
# Move encoding: (from, to, promotion)
# Action space = 64*64*5 = 20480 (promo: none, n, b, r, q)
# ----------------------------
ACTION_DIM = 64 * 64 * 5
PROMO_TO_IDX = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
IDX_TO_PROMO = {v: k for k, v in PROMO_TO_IDX.items()}


def encode_move(m: chess.Move) -> int:
    promo_idx = PROMO_TO_IDX.get(m.promotion, 0)
    return ((m.from_square * 64 + m.to_square) * 5) + promo_idx


def decode_move(idx: int) -> chess.Move:
    base, promo_idx = divmod(idx, 5)
    from_sq, to_sq = divmod(base, 64)
    promo = IDX_TO_PROMO[promo_idx]
    return chess.Move(from_sq, to_sq, promotion=promo)


def legal_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for mv in board.legal_moves:
        mask[encode_move(mv)] = True
    return mask


# ----------------------------
# Board encoding: planes (CNN-friendly)
# 12 piece planes (6 white + 6 black)
# 4 castling rights planes
# 1 side-to-move plane
# 1 en-passant square plane
# 1 rating plane (filled with normalized mover rating)
# => 19 planes total
# ----------------------------
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board, mover_rating: float) -> torch.Tensor:
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    # Pieces
    plane_idx = 0
    for color in (chess.WHITE, chess.BLACK):
        for pt in PIECE_TYPES:
            for sq in board.pieces(pt, color):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                planes[plane_idx, r, c] = 1.0
            plane_idx += 1

    # Castling rights
    planes[12, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # Side to move
    planes[16, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # En passant square (one-hot)
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        planes[17, r, c] = 1.0

    # Rating plane (normalize ~[0,1])
    rating_norm = float(np.clip(mover_rating / 3000.0, 0.0, 1.0))
    planes[18, :, :] = rating_norm

    return torch.from_numpy(planes)


# ----------------------------
# Streaming PGN.zst dataset (IterableDataset)
# ----------------------------
@dataclass
class LichessBand:
    min_rating: int
    max_rating: int


class LichessMoveDataset(IterableDataset):
    """
    Streams games from a .pgn.zst file, yields (board_tensor, legal_mask, move_label).
    Filters to games where BOTH players are within [min_rating, max_rating].

    Note: For maximal correctness and simplicity, use num_workers=0.
    """

    def __init__(
        self,
        pgn_zst_path: str,
        band: LichessBand,
        max_games: Optional[int] = None,
        max_plies_per_game: Optional[int] = None,
        shuffle_buffer: int = 4096,
        seed: int = 0,
        only_standard: bool = True,
    ):
        super().__init__()
        self.path = pgn_zst_path
        self.band = band
        self.max_games = max_games
        self.max_plies_per_game = max_plies_per_game
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.only_standard = only_standard

    def _iter_samples(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        with open(self.path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
                games_kept = 0

                while True:
                    game = chess.pgn.read_game(text)
                    if game is None:
                        break

                    headers = game.headers

                    if self.only_standard:
                        # Skip variants
                        if "Variant" in headers and headers["Variant"] != "Standard":
                            continue

                    # Lichess uses WhiteElo/BlackElo in headers (these are Lichess ratings, not chess.com Elo)
                    try:
                        we = int(headers.get("WhiteElo", "0"))
                        be = int(headers.get("BlackElo", "0"))
                    except ValueError:
                        continue

                    if not (self.band.min_rating <= we <= self.band.max_rating and
                            self.band.min_rating <= be <= self.band.max_rating):
                        continue

                    board = game.board()
                    ply = 0

                    for mv in game.mainline_moves():
                        if self.max_plies_per_game is not None and ply >= self.max_plies_per_game:
                            break

                        mover_rating = we if board.turn == chess.WHITE else be

                        x = board_to_tensor(board, mover_rating)
                        lm = legal_mask(board)
                        y = encode_move(mv)

                        yield x, lm, y

                        board.push(mv)
                        ply += 1

                    games_kept += 1
                    if self.max_games is not None and games_kept >= self.max_games:
                        break

    def __iter__(self):
        rng = random.Random(self.seed + 1337)
        buf = []
        for sample in self._iter_samples():
            buf.append(sample)
            if len(buf) >= self.shuffle_buffer:
                i = rng.randrange(len(buf))
                yield buf.pop(i)
        rng.shuffle(buf)
        yield from buf


# ----------------------------
# Small ResNet policy network
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyNet(nn.Module):
    def __init__(self, in_planes: int = 19, channels: int = 128, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head
        self.p_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 8 * 8, ACTION_DIM)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.flatten(1)
        logits = self.p_fc(p)
        return logits


# ----------------------------
# Checkpoint utils (atomic latest.pt)
# ----------------------------
def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic rename on POSIX


def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, scaler, step: int, args) -> None:
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    atomic_torch_save(ckpt, path)


def load_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, scaler, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "opt" in ckpt and ckpt["opt"] is not None:
        opt.load_state_dict(ckpt["opt"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    # Restore RNGs (best-effort)
    if "torch_rng" in ckpt and ckpt["torch_rng"] is not None:
        torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in ckpt and ckpt["cuda_rng"] is not None:
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        except Exception:
            pass

    return int(ckpt.get("step", 0))


def install_signal_handlers():
    # Let us catch SIGTERM (docker stop) + SIGINT (Ctrl+C) cleanly.
    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def top1_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def mean_entropy_from_logits(logits: torch.Tensor) -> float:
    # logits assumed already masked (illegal = -inf-ish)
    probs = F.softmax(logits, dim=1)
    ent = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
    return float(ent.mean().item())


# ----------------------------
# Training
# ----------------------------
def train(args) -> None:
    install_signal_handlers()

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        # harmless on older versions; nice speed on newer ones
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    ds = LichessMoveDataset(
        pgn_zst_path=args.pgn,
        band=LichessBand(args.min_rating, args.max_rating),
        max_games=args.max_games,
        max_plies_per_game=args.max_plies,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        only_standard=not args.allow_variants,
    )

    if args.num_workers != 0:
        print("WARNING: num_workers>0 with a streaming .pgn.zst iterable dataset is usually a bad idea. "
              "Start with --num-workers 0.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PolicyNet(in_planes=19, channels=args.channels, blocks=args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    # TensorBoard
    writer = None
    if args.log_dir:
        if SummaryWriter is None:
            print("TensorBoard not available (torch.utils.tensorboard import failed). Logging disabled.")
        else:
            writer = SummaryWriter(log_dir=args.log_dir)

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_ckpt(args.resume, model, opt, scaler, device)
        print(f"Resumed from {args.resume} at step {start_step}")

    # Also: if --resume-latest and latest exists, load it
    latest_path = os.path.join(args.out_dir, "latest.pt")
    if args.resume_latest and (not args.resume) and os.path.exists(latest_path):
        start_step = load_ckpt(latest_path, model, opt, scaler, device)
        print(f"Resumed from {latest_path} at step {start_step}")

    model.train()
    opt.zero_grad(set_to_none=True)

    step = start_step
    it = iter(loader)

    # Throughput tracking
    last_t = time.time()
    last_step_for_speed = step

    pbar = tqdm(total=args.steps, desc="train", dynamic_ncols=True)
    pbar.update(step)

    try:
        while step < args.steps:
            try:
                x, lm, y = next(it)
            except StopIteration:
                it = iter(loader)
                continue

            x = x.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                logits = model(x)
                logits = logits.masked_fill(~lm, -1e9)  # mask illegal moves
                loss = F.cross_entropy(logits, y)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            do_opt_step = ((step + 1) % args.grad_accum == 0)
            if do_opt_step:
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            # Metrics (cheap)
            with torch.no_grad():
                acc = top1_acc(logits, y)
                ent = mean_entropy_from_logits(logits)

            # Logging
            if writer is not None and ((step + 1) % args.log_every == 0):
                lr = opt.param_groups[0]["lr"]
                writer.add_scalar("loss/train", float(loss.item()), step + 1)
                writer.add_scalar("acc/train_top1", acc, step + 1)
                writer.add_scalar("entropy/policy", ent, step + 1)
                writer.add_scalar("lr", lr, step + 1)

                now = time.time()
                dt = now - last_t
                if dt > 0.0:
                    steps_done = (step + 1) - last_step_for_speed
                    sps = steps_done / dt
                    writer.add_scalar("throughput/steps_per_sec", sps, step + 1)
                last_t = now
                last_step_for_speed = step + 1

                writer.flush()

            # Checkpointing
            if (step + 1) % args.save_every == 0:
                os.makedirs(args.out_dir, exist_ok=True)
                # Always keep a robust "latest.pt"
                save_ckpt(latest_path, model, opt, scaler, step + 1, args)
                # Optional numbered snapshots
                if args.keep_numbered:
                    snap = os.path.join(args.out_dir, f"policy_step{step+1}.pt")
                    save_ckpt(snap, model, opt, scaler, step + 1, args)

            pbar.set_postfix(loss=float(loss.detach().cpu()), acc=acc, ent=ent)
            pbar.update(1)
            step += 1

    except KeyboardInterrupt:
        print("\nInterrupted. Saving latest checkpoint...")
        os.makedirs(args.out_dir, exist_ok=True)
        save_ckpt(latest_path, model, opt, scaler, step, args)
        print(f"Saved: {latest_path}")
        raise
    finally:
        pbar.close()
        if writer is not None:
            writer.flush()
            writer.close()

    # Final save
    os.makedirs(args.out_dir, exist_ok=True)
    save_ckpt(latest_path, model, opt, scaler, step, args)
    print(f"Done. Final checkpoint: {latest_path}")


# ----------------------------
# Inference helper (sample like a human)
# ----------------------------
@torch.no_grad()
def pick_move(model: nn.Module, board: chess.Board, mover_rating: float, temperature: float = 1.0) -> chess.Move:
    model.eval()
    device = next(model.parameters()).device
    x = board_to_tensor(board, mover_rating).unsqueeze(0).to(device)
    logits = model(x)[0]

    lm = legal_mask(board).to(device)
    logits = logits.masked_fill(~lm, -1e9)

    if temperature <= 0:
        idx = int(torch.argmax(logits).item())
        return decode_move(idx)

    probs = F.softmax(logits / temperature, dim=0)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    mv = decode_move(idx)

    # Safety fallback (promotion edge cases)
    if mv not in board.legal_moves:
        idx = int(torch.argmax(logits).item())
        mv = decode_move(idx)
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="Path to lichess .pgn.zst")

    ap.add_argument("--min-rating", type=int, default=800)
    ap.add_argument("--max-rating", type=int, default=1200)
    ap.add_argument("--allow-variants", action="store_true", help="Do not skip non-standard variants")

    ap.add_argument("--max-games", type=int, default=None, help="Cap number of filtered games (useful for quick tests)")
    ap.add_argument("--max-plies", type=int, default=120, help="Cap plies per game")

    ap.add_argument("--shuffle-buffer", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--grad-accum", type=int, default=1)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)

    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")

    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-numbered", action="store_true", help="Also keep numbered snapshots besides latest.pt")
    ap.add_argument("--out-dir", default="ckpts")

    ap.add_argument("--log-dir", default="runs/chess_imit", help="TensorBoard log dir (empty disables)")
    ap.add_argument("--log-every", type=int, default=50)

    ap.add_argument("--resume", default="", help="Path to checkpoint .pt")
    ap.add_argument("--resume-latest", action="store_true", help="If ckpts/latest.pt exists, resume from it")

    ap.add_argument("--device", default="", help="e.g. cuda, cpu, cuda:0 (default auto)")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended for streaming)")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()