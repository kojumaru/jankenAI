# ✊ Always-Win Janken AI — 絶対に勝つじゃんけんAI

> **An AI that reads your hand motion before you finish — and always plays the winning move.**
> LSTMで手の動きを予測し、出す前に勝ち手を表示するリアルタイムじゃんけんAI。

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00C853?logo=google)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Demo

<!-- ============================================================
     YouTubeにアップロード後、以下の2箇所を書き換えてください:
     1. YOUR_YOUTUBE_VIDEO_ID  → 例: dQw4w9WgXcQ
     2. YOUR_YOUTUBE_VIDEO_ID  → 同じID
     ============================================================ -->
[![Demo Video](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_YOUTUBE_VIDEO_ID)

*Click the image to watch the demo on YouTube.*

---

## How It Works

```
Camera feed
    │
    ▼
MediaPipe Hands  ──▶  21 landmarks (x, y, z)  ×  5 frames
    │
    ▼
LSTM Model  ──▶  Predict: GU / CHOKI / PA
    │
    ▼
Display winning hand before the player finishes!
```

The key insight: **hand motion has a signature before the final pose is formed.**
The LSTM is trained on the *trajectory* of hand landmarks over 5 consecutive frames, not just a snapshot — so it can predict your move mid-gesture.

---

## Features

- **Real-time prediction** at ≈30 fps using webcam
- **LSTM-based sequence model** — understands motion, not just poses
- **MediaPipe hand tracking** — 21 3D landmarks per frame
- **Self-improving loop** — wrong predictions can be corrected and saved as new training data on the fly
- **ASCII art result display** — shows the winning hand (GU / CHOKI / PA) with rhythm sound

---

## Tech Stack

| Layer | Technology |
|---|---|
| Hand Tracking | MediaPipe Hands |
| Sequence Model | PyTorch LSTM |
| Camera / Display | OpenCV |
| Audio | Pygame |
| Data Processing | NumPy |

### Model Architecture

```
Input:  (batch, 5 frames, 63 features)   ← 21 landmarks × (x,y,z)
         │
    LSTM (hidden=32, layers=1)
         │
    Linear (32 → 3)
         │
Output: GU / CHOKI / PA
```

Preprocessing normalizes landmarks relative to the wrist (landmark 0) and scales by the distance to the middle finger base (landmark 9), making predictions robust to hand size and position.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam

### Installation

```bash
git clone https://github.com/kojumaru/always-win-janken.git
cd always-win-janken
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Start a round (plays "Janken PON!" sound) |
| `SPACE` | Confirm prediction was correct |
| `V` | Correct label: GU (and save as training data) |
| `B` | Correct label: CHOKI (and save as training data) |
| `N` | Correct label: PA (and save as training data) |
| `Q` | Quit |

---

## Project Structure

```
always-win-janken/
├── main.py              # Main application (game loop + inference)
├── cleaningData.py      # Training data labeling & cleaning tool
├── janken_EF.pth        # Trained LSTM model weights
├── jankenpon_rhythm.wav # Game sound effect
└── requirements.txt
```

---

## Background

Built this project to explore **real-time gesture recognition** beyond simple pose classification.
The challenge was not "can the model classify a still hand?" but
**"can it read the intent before the gesture is complete?"**

The answer: yes — the LSTM picks up on finger curl velocity and trajectory patterns that differ between GU, CHOKI, and PA even in the first moments of motion.

---

## Author

**kojumaru** — [GitHub](https://github.com/kojumaru)
