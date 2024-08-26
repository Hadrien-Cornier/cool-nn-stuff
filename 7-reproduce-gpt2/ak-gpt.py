import torch
from torchinfo import summary
from torch.nn import functional as F
from dataclasses import dataclass
import os

import torch.nn as nn
torch.manual_seed(1337)

print(os.getcwd())
input_path = os.path.abspath(os.getcwd()+"/7-reproduce-gpt2/data/shakespeare.txt")

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab :\n" , ''.join(chars))
print("vocab size :", vocab_size)

# simple tokenizer = convert each character to its index in the vocab
#string to index and index to string
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

@dataclass
class Config:
    batch_size = 32
    block_size = 16
    vocab_size = vocab_size
    learning_rate = 1e-2
    max_iters = 500
    eval_iters = 20
    n_embd = 32
    n_layer = 8
    n_head = 4
    dropout = 0.5
    T = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property 
    def head_size(self): return self.n_embd // self.n_head ;

config = Config()

data = encode(text)
n = int(len(data)*0.9)
train_data = torch.tensor(data[:n], dtype=torch.long)
val_data = torch.tensor(data[n:], dtype=torch.long)

print("first training row of the training data:")
print(train_data[:config.block_size])

def get_batch(split):
    block_size = config.block_size
    batch_size = config.batch_size
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.ffwd = FeedForward(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.sa_heads = MultiHeadAttention(4, config.n_embd//4)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)], LayerNorm(config.n_embd))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # unembedding layer that takes hidden state and returns probabilities for each token

        if targets is None:
            loss = None
        else:
            # batch_size x sequence_length x vocab_size
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -config.block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits/config.T, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Head(nn.Module):
    #Attention Head
    def __init__(self, head_size=12):
        super().__init__()
        self.scale = 1 / (head_size ** 0.5)
        self.key = nn.Linear(config.n_embd,head_size)
        self.query = nn.Linear(config.n_embd,head_size)
        self.value = nn.Linear(config.n_embd,head_size)

    def forward(self, x, mask=None):
        # batch_size x sequence_length x head_size
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)

        # BSH,BHS -> BSS
        # batch_size x sequence_length x sequence_length
        # this is the attention map
        # we could represent this in einstain notation in jax like this : bth,bht->btt
        wei = q @ k.transpose(-2, -1) * self.scale

        if mask is not None:
            # For causal masking the mask would be a lower triangular matrix of shape (sequence_length, sequence_length)
            wei = wei.masked_fill(mask == 0, float('-inf'))

        # These are the attnetion scores for each token, ie where do I distribute my attention ?
        # B x S x S
        wei = F.softmax(wei, dim=-1)

        # # BSS,BSH -> BSH 
        # SO we start with BHS and return to BSH
        # the attention scores give the weights in the interpolation of the values of other tokens
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, head_size=12, num_heads=12) :
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dp = Dropout(config.dropout)

    def forward(self, x):
        return self.dp(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class Dropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.empty_like(x).bernoulli_(1 - self.p)
            return x * mask / (1 - self.p)
        return x

class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(n_embd))
        self.b = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.g * (x - mean) / (std + self.eps) + self.b
    
class FeedForward(nn.Module):  
    def __init__(self, n_embd):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(config.dropout),
        )
    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config.head_size, config.n_head)
        self.ffwd = FeedForward(config.n_embd)
        self.ln1 = LayerNorm(config.n_embd)
        self.ln2 = LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

model = GPTLanguageModel()
model.to(config.device)
print(summary(model))
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for steps in range(config.max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    if steps%100==0:
      print(" at step ", steps, "trained on ", str((steps+1)*config.batch_size), ", loss is " , estimate_loss() )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))