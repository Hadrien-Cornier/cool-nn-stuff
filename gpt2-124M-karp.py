from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F

class GPT2_124M_Karp:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.model.to('cuda')

    def generate(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# dataclass automatically generates 
#  __init__, __repr__, __eq__, and __hash__ methods
@dataclass
class GPTConfig:
    block_size = 256
    vocab_size = 50257
    n_embd = 768
    n_layer = 12
    n_head = 12

class GPT(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config


        # self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        # self.positions_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # self.ln_f = nn.LayerNorm(config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd)),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            # h represents the layers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

# ModuleList and ModuleDict store model components 
# and ModuleList is indexed with a number 
# and ModuleDict is indexed with a string key

class c(nn.Module):

    """We also increase the context size from 512 to 1024 tokens """
    def __init__(self, config):
        super().__init__()
        """Layer normalization (Ba et al., 2016) was moved to the input of each sub-block"""
        self.ln_1 = nn.LayerNorm(config.n_embd)
        """an additional layer normalization was added after the final selfattention block"""
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
        )
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head)
        self.n_embd = config.n_embd

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), x, x, attn_mask=mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x




# let's reproduce the transformer block


