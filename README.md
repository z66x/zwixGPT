# zwixGPT

> *i made an AI that raps. it's not good. it's also not bad. it's something.*

---

## what is this

a transformer i built from scratch that generates text character by character. no HuggingFace, no shortcuts, no dignity.

trained it on shakespeare first (as you do), then on kanye lyrics because that felt like the logical next step.

---

## does it work

kind of. here's what it outputs after ~4000 steps on kanye lyrics:

```
Ug, Fucking right didamant
I faking after fuck in hurts niggas
This eyes fadiculous
'Cause to life)
Tryin' conturifuckin' you, I'll it
I'm not remember who crisechi gon' 're money (b
```

so yes. it captures the vibe. the words are fake but the energy is real.

---

## architecture (the short version)

decoder-only transformer. causal self-attention so it can't cheat and look ahead. residual connections so gradients don't die. pre-layernorm so training doesn't explode in my face.

that's it. that's the whole thing.

---

## experiment log

### exp 0 — shakespeare
just vibing, establishing a baseline.

| thing | value |
|---|---|
| dataset | tinyshakespeare (1.1MB) |
| vocab | 65 chars |
| steps | 5000 |
| final train loss | 0.9443 |
| final test loss | 1.5309 |
| weight file | 50.24 MB (github yelled at me) |

### exp 1 — kanye
the real one. took 4 runs to get right.

| thing | value |
|---|---|
| dataset | kanye lyrics (~150-250K chars) |
| vocab | 95 chars |
| steps | 6000 |
| final test loss | 1.68 |
| weight file | ~19 MB (github chilled out) |

**the dropout saga** — ran it 4 times because i kept getting it wrong:

| run | dropout | what happened |
|---|---|---|
| 1 | 0.33 | decent, gap widening |
| 2 | 0.40 | model had a stroke, output was `?AAAAAAAAHHHHHHHHHHH!!` |
| 3 | 0.36 | no overfitting but no real meaning in output |
| 4 | 0.4 + n_layer=4 + 6000 steps | looks good. still on the hunt for better tuning |

---

## final hyperparams

```python
batch_size = 32
block_size = 128
n_emb = 300
n_head = 6
dropout_rate = 0.4
n_layer = 4
max_steps = 6000
learning_rate = 3e-4
```

---

## setup

```bash
git clone https://github.com/<your-username>/zwixGPT.git
cd zwixGPT
pip install torch
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python zwixGPT.py
```

swap `input.txt` for any `.txt` corpus and it'll learn to write like that instead. probably.

---

## files

```
zwixGPT/
├── zwixGPT.py                    # the whole model
├── bigram-model.ipynb            # where it started
├── shakespeare_from_temu.pth     # exp 0 weights
└── kanye_from_temu.pth           # exp 1 weights
```

---

*built from scratch. runs on a T4. named after myself.*