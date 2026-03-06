# Chess Imitation Training

This repo contains a small PyTorch project for training a **human-move imitation chess model** from Lichess PGN data, then testing it interactively from the terminal.

The goal is **not** to train a superhuman engine. The goal is to train a model that predicts the kinds of moves humans in a given rating band tend to play.

That means you can do things like:
* Train on lower-rated Lichess games.
* Get a model that plays more like club players or scrubs.
* Practice against a style closer to the games you actually face.

---

## What is in this repo?

**Main scripts:**
* `chess_imitation.py`: Training script. Streams `.pgn.zst` Lichess data, builds training examples, trains a policy network, logs metrics, and saves checkpoints.
* `play_chess_imitation.py`: Interactive terminal app for playing against a trained checkpoint.
* `Dockerfile`: CUDA-enabled container image for running training/testing inside Docker.

**Optional output directories:**
* `ckpts/` — model checkpoints
* `runs/` — TensorBoard logs
* `data/` — Where your `lichess_db_standard_rated_YYYY-MM.pgn.zst` files live

---

## Requirements

Host system assumptions:
* Ubuntu 24.04 host
* NVIDIA GPU (example: RTX 2080 Ti)
* NVIDIA driver installed and working on the host
* Docker installed
* NVIDIA Container Toolkit installed and configured

---

## Folder Layout

Suggested layout:

```text
chess-train/
  Dockerfile
  chess_imitation.py
  play_chess_imitation.py
  ckpts/
  runs/
  data/
