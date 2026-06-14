import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(66)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# hyper-parameters
batch_size = 64
block_size = 256
n_emb = 384
n_head = 6
dropout_rate = 0.2
n_layer = 6
eval_interval = 500
eval_iters = 66
max_steps = 5000
learning_rate = 6e-4

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# extracting unique characters
print("chars in dataset: ", len(text))
chars = list(set(text))
voc_size = len(chars)
print(voc_size, chars)

# creating mapping
ctoi = {ch:i for i,ch in enumerate(chars)}
itoc = {i:ch for i,ch in enumerate(chars)}

# encode / decode strings
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])

# using torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

# spliting into train and test data
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# one head self attention
class head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_emb)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x +self.ffwd(self.ln2(x))
        return x

class BigramLangModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(voc_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, voc_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        final_emb = tok_emb + pos_emb
        final_emb = self.blocks(final_emb)
        final_emb = self.ln_f(final_emb)
        logits = self.lm_head(final_emb)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_trunc = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_trunc)
            # we need the embedding of the last step only
            logits = logits[:, -1, :] # (B, C)
            # we convert the predictions into probabilties
            probs = F.softmax(logits, dim=-1) # (B, C)
            # we roll a dies based on probabities to get next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # we add the token we got to idx
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        return idx

model = BigramLangModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("model initialized and loaded, roger that!")

for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'shakespeare_from_temu.pth')
print("cargo Secured, neural state extracted and locked as 'shakespeare_from_temu.pth'!")

print("shakepeare starts to yap...\n")
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(idx = context, max_new_tokens = 660)[0].tolist()))