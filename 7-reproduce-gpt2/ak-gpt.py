import torch
from torchinfo import summary
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
torch.manual_seed(1337)

# print pwd
import os
print(os.getcwd())
input_path = os.path.abspath(os.getcwd()+"/7-reproduce-gpt2/data/shakespeare.txt")
# read it in to inspect it
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab :\n" , ''.join(chars))
print("vocab size :", vocab_size)


# hyperparams 
@dataclass
class Config:
    batch_size = 32
    block_size = 8
    vocab_size = vocab_size
    n_embd = 16
    n_layer = 1
    n_head = 4 # number of heads

    @property 
    def head_size(self): return self.n_embd // self.n_head ;
    # we split our hidden representation into multiple independednt channels 
    #that each carry their own stream of information

config = Config()

# print pwd
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocab :\n" , ''.join(chars))
print("vocab size :", vocab_size)

# TOKENIZER 

# create a mapping from characters to integers
# most basic tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))
# hyperparams 
# no learning can happen if we do this

# Dataloader : 
data = encode(text)
n = int(len(data)*0.9) # train set is 90% of the data
train_data = torch.tensor(data[:n], dtype=torch.long)
val_data = torch.tensor(data[n:], dtype=torch.long)

print("first training row of the training data:")
print(train_data[:config.block_size])

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    block_size = config.block_size
    batch_size = config.batch_size
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # batch
    # y_i = x_i + 1 
    # batch : B elements : | features : [x_1,...,x_block_size] | target : [y_1,...,y_block_size]
    return x, y

# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

# for b in range(config.batch_size): # batch dimension
#     for t in range(config.block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")
# the bigram language model just predicts the 
# next word with a lookup table of frequencies
# but we can also sample from the implied softmax distribution
class SimpleBigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

# we just need to populate self.token_embedding_table with the occurence counts and we have the distribution lookup for every token to the next
        # idx and targets are both (B,T) tensor of integers
        # C = classes = vocabulary size in our bigram case
        # B = training batch size
        # T = context length
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # this is exactly what the end of the gpt2 model will be later
            # just use the distribution probability of the next token and sample from it
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class BigramLanguageModel(nn.Module):
    #we are improving this bit by bit by adding attenstion layer inside it 
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.ffwd = FeedForward(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.sa_heads = MultiHeadAttention(4, config.n_embd//4)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

    def forward(self, idx, targets=None):
        # we just need to populate self.token_embedding_table with 
        # the occurence counts and we have the distribution lookup for every token to the next
        # idx and targets are both (B,T) tensor of integers
        # C = classes = vocabulary size in our bigram case
        # B = training batch size
        # T = context length
        B = idx.shape[0]
        T = idx.shape[1]
        # print("idx shape", idx.shape)
        tok_emb = self.token_embedding_table(idx) # (B,T)
        # print("tok_emb shape", tok_emb.shape)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb
    
        x = self.sa_heads(x) # collect context
        x = self.ffwd(x) # think based on context
        logits = self.lm_head(x) # decodes words from the latent representation by sampling
 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # this is exactly what the end of the gpt2 model will be later
            # just use the distribution probability of the next token and sample from it
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size=12):
        super().__init__()
        # n_embed = C
        self.scale = 1 / (head_size ** 0.5)
        # What is this about ? 

        # this is version 1 of the head
        self.key = nn.Linear(head_size,head_size)
        self.query = nn.Linear(head_size,head_size)
        self.value = nn.Linear(head_size,head_size)

        # this is version 2 of the head, we stack the key,value,pair
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, mask=None):
        # kqv = self.c_attn(x).view(B, T, 3, config.n_embd)
        k=self.key(x) # (B, T, C) -> k.transpose(-2, -1) = (B, C, T)
        q=self.query(x) # (B, T, C)
        v=self.value(x) # (B, T, C)

        # # (B, T, C)
        wei = q @ k.transpose(-2, -1) * self.scale

        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1) # (B, T, T)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size=12, num_heads=12) :
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        """ multi-headed is just single head in parallel 
        in different channels for the same input sequence"""
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module): 
    def __init__(self, n_embd):
        super().__init__()
        # now they think on the data individually
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )
    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config.head_size, config.n_head)
        self.ffwd = FeedForward(config.n_embd)

    def forward(self, x):
        x = self.attn(x) # collect context
        x = self.ffwd(x) # think based on context
        return x

model = BigramLanguageModel()
eval_iters = 100
print(summary(model))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
print(f"Training on {len(train_data)} examples")
#
for steps in range(1000): # increase number of steps for good results...
    
    # sample a batch of data
    xb, yb = get_batch('train')
    # print("xb shape", xb.shape)
    # print("yb shape", yb.shape)
    # print("xb", yb)

    # evaluate the loss
    logits, loss = model(xb, yb)
    if steps%100==0:
      print(" at step ", steps, "trained on ", str((steps+1)*config.batch_size), ", loss is " , estimate_loss() )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())