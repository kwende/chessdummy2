cat << 'EOF' > README.md
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

chess-train/
  Dockerfile
  chess_imitation.py
  play_chess_imitation.py
  ckpts/
  runs/
  data/

---

## 1. Verify the Host GPU Works

Before touching Docker, make sure the host can see the GPU:
nvidia-smi

> **Note:** If that fails, stop there and fix the host NVIDIA driver first.

---

## 2. If Docker GPU Support is Broken

If this command fails:
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

With an error like: `could not select device driver "" with capabilities: [[gpu]]`, that usually means the NVIDIA Container Toolkit is missing or not configured.

**Install NVIDIA Container Toolkit:**
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

**Configure Docker to use the NVIDIA runtime:**
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

**Verify GPU access inside Docker:**
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

Then retry:
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

**Useful debugging commands:**
which nvidia-ctk
dpkg -l | grep -E 'nvidia-container|libnvidia-container'
cat /etc/docker/daemon.json
docker info | grep -i -E 'runtimes|default runtime'

---

## 3. Build the Docker Image

From the repo directory:
docker build -t chess-imit:cuda121 .

---

## 4. Dockerfile

This is the `Dockerfile` currently expected by the instructions here:

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      ca-certificates \
      zstd \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --no-cache-dir \
      python-chess zstandard numpy tqdm tensorboard

WORKDIR /workspace

CMD ["bash"]

---

## 5. Verify the Container Can See the GPU

Sanity check:
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

If that works, the container GPU plumbing is alive and not full of demons.

---

## 6. Training Data

The training script expects a Lichess `.pgn.zst` file.

> **Important:** Do not decompress it first. The training code streams `.pgn.zst` directly.

**Example file:** `lichess_db_standard_rated_2026-01.pgn.zst`

Mount your host data directory into the container, for example at `/data`.

---

## 7. Training the Model

**Basic training run:**
From the repo directory:
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  -v "$(pwd)"/ckpts:/workspace/ckpts \
  -v "$(pwd)"/runs:/workspace/runs \
  -v /absolute/path/to/data:/data \
  chess-imit:cuda121 \
  python3 /workspace/chess_imitation.py \
    --pgn /data/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --min-rating 800 --max-rating 1200 \
    --max-games 5000 \
    --steps 20000 \
    --batch-size 128 \
    --amp \
    --out-dir /workspace/ckpts \
    --save-every 500 \
    --keep-numbered

**This command does the following:**
* Mounts your repo into `/workspace`
* Mounts host checkpoints into `/workspace/ckpts`
* Mounts TensorBoard logs into `/workspace/runs`
* Mounts training data into `/data`
* Trains on games where both players are between 800 and 1200
* Saves checkpoints every 500 steps
* Keeps `latest.pt` plus numbered snapshots

---

## 8. Important Training Options

* **Required Input:** `--pgn /data/file.pgn.zst` (Path to the Lichess dataset)
* **Rating Band:** `--min-rating 800` | `--max-rating 1200` (Only keeps games where both players fall in this band)
* **Training Length:** `--steps 20000` (Number of training steps)
* **Limit Games:** `--max-games 5000` (Useful for quick experiments instead of chewing through a giant dataset forever)
* **Limit Plies:** `--max-plies 120` (Caps the number of plies taken from each game)
* **Batch Size:** `--batch-size 128` (Adjust based on VRAM)
* **Mixed Precision:** `--amp` (Recommended on CUDA)
* **Checkpoint Frequency:** `--save-every 500` (Saves `latest.pt` every 500 steps)
* **Keep Snapshots:** `--keep-numbered` (Keeps files like `policy_step500.pt` in addition to `latest.pt`)
* **Output Directory:** `--out-dir /workspace/ckpts` (Where checkpoints go)
* **TensorBoard Logs:** `--log-dir /workspace/runs/chess_imit`
* **Resume Explicitly:** `--resume /workspace/ckpts/latest.pt`
* **Resume Automatically:** `--resume-latest` (If `/workspace/ckpts/latest.pt` exists, it resumes from there)
* **DataLoader Workers:** `--num-workers 0` (Recommended for the current streaming iterable dataset. Do not get clever here immediately. Cleverness often becomes archaeology.)

---

## 9. Example Training Commands

**Quick smoke test:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  -v "$(pwd)"/ckpts:/workspace/ckpts \
  -v "$(pwd)"/runs:/workspace/runs \
  -v /absolute/path/to/data:/data \
  chess-imit:cuda121 \
  python3 /workspace/chess_imitation.py \
    --pgn /data/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --min-rating 800 --max-rating 1200 \
    --max-games 500 \
    --steps 500 \
    --batch-size 64 \
    --amp \
    --out-dir /workspace/ckpts \
    --save-every 100 \
    --keep-numbered

**Resume training from latest:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  -v "$(pwd)"/ckpts:/workspace/ckpts \
  -v "$(pwd)"/runs:/workspace/runs \
  -v /absolute/path/to/data:/data \
  chess-imit:cuda121 \
  python3 /workspace/chess_imitation.py \
    --pgn /data/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --min-rating 800 --max-rating 1200 \
    --steps 20000 \
    --batch-size 128 \
    --amp \
    --out-dir /workspace/ckpts \
    --resume-latest

---

## 10. What the Training Script Does

`chess_imitation.py` currently:
* Streams a `.pgn.zst` file directly
* Parses PGN games with `python-chess`
* Filters games by rating band
* Converts each position into a tensor of board planes
* Predicts the move a human played from that position
* Uses a policy network with residual blocks
* Masks illegal moves during training and inference
* Logs training metrics to TensorBoard
* Saves checkpoints
* Supports clean resume
* Saves on interrupt (Ctrl+C / SIGTERM)

**The model is trying to learn:** `P(move | position, rating band)`... not perfect chess.

---

## 11. About the AMP Warnings

You may see API deprecation warnings like this during training:
FutureWarning: torch.cuda.amp.GradScaler(...) is deprecated
FutureWarning: torch.cuda.amp.autocast(...) is deprecated

Training still works, but to modernize the code, change:
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

And change:
with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):

---

## 12. Running TensorBoard

In another terminal:
docker run --rm -it -p 6006:6006 \
  -v "$(pwd)"/runs:/runs \
  chess-imit:cuda121 \
  tensorboard --logdir /runs --host 0.0.0.0 --port 6006

Then open: http://localhost:6006

Metrics logged include: training loss, top-1 accuracy, policy entropy, learning rate, and throughput.

---

## 13. Playing Against the Model

`play_chess_imitation.py` is a terminal app that:
* Renders the board as ASCII
* Prompts you for your move
* Accepts SAN or UCI notation
* Shows the model’s top candidate moves
* Samples and plays the model move
* Repeats until game over

**Example Run:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  chess-imit:cuda121 \
  python3 /workspace/play_chess_imitation.py \
    --ckpt /workspace/ckpts/latest.pt

**Play as Black:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  chess-imit:cuda121 \
  python3 /workspace/play_chess_imitation.py \
    --ckpt /workspace/ckpts/latest.pt \
    --color black

**Force deterministic play:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  chess-imit:cuda121 \
  python3 /workspace/play_chess_imitation.py \
    --ckpt /workspace/ckpts/latest.pt \
    --temperature 0

**Make the model more chaotic:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  chess-imit:cuda121 \
  python3 /workspace/play_chess_imitation.py \
    --ckpt /workspace/ckpts/latest.pt \
    --temperature 1.2

---

## 14. Play Script Options

* **Required Checkpoint:** `--ckpt /workspace/ckpts/latest.pt`
* **Human Side:** `--color white` or `--color black`
* **Sampling Temperature:** `--temperature 0.8` (Lower = more deterministic, Higher = more varied/weirder)
* **Show Top K Moves:** `--topk 5` (Shows top candidate moves and probabilities)
* **Override Model Rating:** `--model-rating 1000` (If omitted, uses the midpoint of the checkpoint’s training band)
* **Start from Custom FEN:** `--start-fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
* **Force Device:** `--device cuda` or `--device cpu`

---

## 15. Commands Inside the Play App

Once inside the interactive prompt, you can use the following commands:
`help`, `board`, `fen`, `top`, `undo`, `quit`

Moves can be entered as:
* **SAN:** `e4`, `Nf3`, `Qxe5+`
* **UCI:** `e2e4`

---

## 16. Example Interaction

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Turn: White

Your move > e4

Top 5 model candidates:
   1. e5           e7e5    p=0.3124
   2. c5           c7c5    p=0.2011
   3. e6           e7e6    p=0.1180
   4. d5           d7d5    p=0.0943
   5. Nf6          g8f6    p=0.0721

Model plays: c5 (c7c5)

---

## 17. Checkpoints

The training script saves:
* `latest.pt` — the newest checkpoint
* Optional numbered snapshots if `--keep-numbered` is used

A checkpoint includes: model weights, optimizer state, AMP scaler state, current step, saved args, and RNG state. This allows proper resume instead of fake resume theater.

---

## 18. Suggested First Workflow

**1. Verify host and Docker GPU:**
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi

**2. Build the image:**
docker build -t chess-imit:cuda121 .

**3. Run a tiny smoke-test training job:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  -v "$(pwd)"/ckpts:/workspace/ckpts \
  -v "$(pwd)"/runs:/workspace/runs \
  -v /absolute/path/to/data:/data \
  chess-imit:cuda121 \
  python3 /workspace/chess_imitation.py \
    --pgn /data/lichess_db_standard_rated_YYYY-MM.pgn.zst \
    --min-rating 800 --max-rating 1200 \
    --max-games 500 \
    --steps 500 \
    --batch-size 64 \
    --amp \
    --out-dir /workspace/ckpts \
    --save-every 100 \
    --keep-numbered

**4. Play against the checkpoint:**
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  chess-imit:cuda121 \
  python3 /workspace/play_chess_imitation.py \
    --ckpt /workspace/ckpts/latest.pt
EOF
