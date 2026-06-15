# zwixGPT

A character-level language model built from scratch in PyTorch — no frameworks, no abstractions. Trained on Shakespeare to establish a baseline, then on Kanye West lyrics for domain transfer.

---

## Architecture

Decoder-only transformer with causal self-attention. The full stack:

- **Causal self-attention** with scaled dot-product (`QKᵀ / √dₖ`) and a lower-triangular mask to prevent the model from looking ahead
- **Multi-head attention** — parallel heads concatenated and projected back to embedding space, each attending to different representational subspaces
- **Position-wise FFN** with 4× hidden expansion: `n_emb → 4·n_emb → n_emb`
- **Pre-LayerNorm** applied before each sub-block — keeps gradients stable at depth without the Post-LN blowup problem
- **Residual connections** around both sub-blocks per layer for clean backprop flow

---

## Experiments

### Exp 0 — Shakespeare

```python
batch_size = 64
block_size = 256
n_emb = 384
n_head = 6        # 384 / 6 = 64 dims per head
dropout_rate = 0.2
n_layer = 6
max_steps = 5000
learning_rate = 6e-4
```

| Step | Train | Test |
|---|---|---|
| 0 | 4.4070 | 4.4051 |
| 500 | 1.8236 | 1.9598 |
| 1000 | 1.4429 | 1.6399 |
| 2000 | 1.2211 | 1.5096 |
| 2500 | 1.1530 | **1.4833** |
| 3000 | 1.1016 | 1.4845 |
| 4500 | 0.9443 | 1.5309 |

Initial loss of 4.4070 aligns with the theoretical bound `ln(65) ≈ 4.17` for uniform random prediction over a 65-character vocabulary. Test loss bottomed at **1.4833 around step 2500** before creeping back up — the model had enough capacity (`n_layer=6`, `n_emb=384`) for the 1.1MB corpus.

Weight file: `shakespeare_from_temu.pth` — 50.2 MB.

---

### Exp 1 — Kanye Lyrics

Smaller corpus (183,627 chars, 95 unique characters) — architecture scaled down accordingly.

```python
batch_size = 32    # halved; smaller corpus needs more gradient diversity per epoch
block_size = 128   # halved; matches lyric line length, reduces memory footprint
n_emb = 300        # scaled down from 384; divisible by n_head (300 / 6 = 50 dims per head)
n_head = 6
dropout_rate = 0.4 # higher than Exp 0; counteracts overfitting on repetitive bar/hook structure
n_layer = 4        # scaled down from 6; dataset too small to fill 6-layer capacity
max_steps = 6000   # extended; test loss still declining past step 4000
learning_rate = 3e-4  # halved from 6e-4; conservative for small dataset
```

**Why these differ from Exp 0:**

The Shakespeare corpus is ~6× larger with more structural variety. That justifies deeper architecture (`n_layer=6`), larger batches, wider context, and lower dropout. The Kanye dataset at 183K chars doesn't have enough signal to fill a 6-layer network — excess capacity just gets used to memorize training sequences instead of generalizing. Every knob was turned down to match the data, except dropout which went up to compensate for the repetitive lyric structure.

| Step | Train | Test |
|---|---|---|
| 0 | 4.6967 | 4.7058 |
| 500 | 2.2916 | 2.3209 |
| 1000 | 1.8748 | 1.9596 |
| 2000 | 1.5438 | 1.7837 |
| 3000 | 1.3396 | 1.6924 |
| 3500 | 1.2717 | **1.6760** |
| 4000 | 1.2021 | 1.6893 |
| 5500 | 1.0249 | 1.6761 |

Test loss bottomed at **1.6760 at step 3500** and plateaued — training beyond ~4000 steps is diminishing returns. Initial loss of 4.6967 matches `ln(95) ≈ 4.55` for the 95-character vocabulary.

Weight file: `kanye_from_temu.pth` — 18.2 MB.

---

## Setup

```bash
git clone https://github.com/z66x/zwixGPT.git
cd zwixGPT
pip install torch
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python zwixGPT.py
```

Swap `input.txt` for any `.txt` corpus to train on a different domain. The model auto-detects CUDA — T4 or better recommended for full runs.

---

## Files

```
zwixGPT/
├── zwixGPT.py                              # model architecture + training loop
├── bigram-model.ipynb                      # bigram baseline — tokenization & batching
├── zwixGPT_trained_on_shakespeare_text.ipynb  # Exp 0 training notebook
├── zwixGPT_trained_on_kanye_west.ipynb     # Exp 1 training notebook
├── kanye_from_temu.pth                     # Exp 1 weights
└── kanye_west_lyrics.txt                   # Exp 1 dataset
```

---

*Built from scratch. No HuggingFace. No shortcuts.*