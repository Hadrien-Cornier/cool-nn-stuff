import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softmax
from flax.linen import summary

# Simple Jax Transformer implementation
class AttentionHead(nn.Module):
    hidden_dim: int = 768  # D
    head_size: int = 12 # H

    def setup(self):
        self.scale = 1 / (self.head_size ** 0.5)
        self.key_weight = self.param('key_weight', jax.random.normal, (self.hidden_dim, self.head_size))
        self.query_weight = self.param('query_weight', jax.random.normal, (self.hidden_dim, self.head_size))
        self.value_weight = self.param('value_weight', jax.random.normal, (self.hidden_dim, self.head_size))


    def __call__(self, x, mask=None):
        """
        Apply self-attention mechanism to the input tensor.
        Args:
            x (jax.numpy.ndarray): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask (jax.numpy.ndarray, optional): Mask tensor of shape (batch_size, sequence_length) with 0s indicating positions to be masked. Defaults to None.
        Returns:
            jax.numpy.ndarray: Output tensor of shape (batch_size, sequence_length, head_size).        
        """
        # the signature for this call is (BSD -> BSH)
        # batch_size : B
        # sequence_length: S
        # hidden_dim : D
        # head_size : H

        # print(x.shape, self.key_weight.shape, self.query_weight.shape, self.value_weight.shape)
        k = jnp.einsum('BSD,DH->BSH', x, self.key_weight)
        q = jnp.einsum('BSD,DH->BSH', x, self.query_weight)
        v = jnp.einsum('BSD,DH->BSH', x, self.value_weight)

        # k,q,v shape: (batch_size), sequence_length, head_size)
        # print(k.shape, q.shape, v.shape)
        kt = k.transpose(0, 2, 1)
        # print(q.shape, "x", kt.shape)
        # here T = S, but we need it because otherwise einsum will not work
        attention_map = jnp.einsum('BSH,BHT->BST', q, kt) * self.scale

        if mask is not None:
            attention_map = jnp.where(mask == 0, float('-inf'), attention_map)

        attention_scores = softmax(attention_map, axis=-1)

        out = jnp.einsum('BSS,BSH->BSH', attention_scores, v)
        return out

def test_attention_head():
    rng = jax.random.PRNGKey(0)
    B, S, D, H = 16, 10, 768, 12
    head = AttentionHead(hidden_dim=D,head_size=H)
    x = jax.random.normal(rng, (B, S, D))
    mask = None
    params = head.init(rng, x, mask)
    out = head.apply(params, x, mask)
    assert out.shape == (B, S, H)

class Dropout(nn.Module):
    rate: float = 0.1

    def setup(self):
        self.rng = jax.random.PRNGKey(0)
        self.dropout_rate = self.rate

    def __call__(self, x, deterministic: bool = True):
        if deterministic or self.dropout_rate == 0:
            return x
        # sample from rng for the dropout
        keep = jax.random.bernoulli(self.rng, 1.0 - self.dropout_rate, x.shape)
        x = jnp.where(keep, x / (1.0 - self.dropout_rate), 0)
        return x
    
class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention class represents a multi-head attention mechanism.
    Attributes:
        hidden_dim (int): The hidden dimension size (D) of the attention mechanism.
        head_size (int): The size (H) of each attention head.
        num_heads (int): The number of attention heads (D/H).
        dropout_rate (float): The dropout rate for the attention mechanism.
    Methods:
        __call__(x, deterministic=True):
            Applies the multi-head attention mechanism to the input tensor x.
            Args:
                x (ndarray): The input tensor of shape (B, S, D).
                deterministic (bool, optional): Whether to apply dropout deterministically. Defaults to True.
            Returns:
                ndarray: The output tensor after applying the multi-head attention mechanism, of shape (B, S, D).
    """
    hidden_dim: int = 768 # D
    head_size: int = 12 # H
    num_heads: int = 64 # D/H
    dropout_rate: float = 0.1

    def setup(self):
        self.heads = [AttentionHead(hidden_dim=self.hidden_dim, head_size=self.head_size) for _ in range(self.num_heads)]
        self.proj = self.param('proj', jax.random.normal, (self.hidden_dim, self.hidden_dim))
        self.dp = Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: bool = True):
        # h(x) = BSH
        heads_output = [h(x) for h in self.heads]
        # concatenated = BSD
        concatenated = jnp.concatenate(heads_output, axis=-1)
        projected = jnp.einsum('BSD,DD->BSD', concatenated, self.proj)
        return self.dp(projected, deterministic=deterministic)

def test_multi_head_attention():
    rng = jax.random.PRNGKey(0)
    B, S, D, H = 16, 10, 768, 12
    mha = MultiHeadAttention(hidden_dim=D, head_size=H, num_heads=D//H)
    x = jax.random.normal(rng, (B, S, D))
    params = mha.init(rng, x)
    out = mha.apply(params, x)
    assert out.shape == (B, S, D)

class LayerNorm(nn.Module):
    hidden_dim: int # D
    eps: float = 1e-5

    def setup(self):
        self.g = self.param('g', jax.nn.initializers.ones, (self.hidden_dim,))
        self.b = self.param('b', jax.nn.initializers.zeros, (self.hidden_dim,))

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        std = jnp.std(x, axis=-1, keepdims=True)
        return self.g * (x - mean) / (std + self.eps) + self.b

class FeedForward(nn.Module):
    hidden_dim: int # D
    dropout_rate: float = 0.1

    def setup(self):
        self.fc1 = self.param('fc1', jax.random.normal, (self.hidden_dim, 4*self.hidden_dim))
        self.fc2 = self.param('fc2', jax.random.normal, (4*self.hidden_dim, self.hidden_dim))
        self.dropout = Dropout(rate=self.dropout_rate)

    def __call__(self, x, deterministic: bool = True):
        # E = 4D
        x = jnp.einsum('BSD,DE->BSE', x, self.fc1)
        x = jax.nn.gelu(x)
        x = jnp.einsum('BSE,ED->BSD', x, self.fc2)
        return self.dropout(x, deterministic=deterministic)

class Block(nn.Module):
    hidden_dim: int = 768 # D
    head_size: int = 12 # H
    num_heads: int = 64 # D/H
    dropout_rate: float = 0.1

    def setup(self):
        self.attn = MultiHeadAttention(hidden_dim=self.hidden_dim, head_size=self.head_size, num_heads=self.num_heads, dropout_rate=self.dropout_rate)
        self.ffwd = FeedForward(hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate)
        self.ln1 = LayerNorm(hidden_dim=self.hidden_dim)
        self.ln2 = LayerNorm(hidden_dim=self.hidden_dim)

    def __call__(self, x, deterministic: bool = True):
        x = x + self.attn(self.ln1(x), deterministic=deterministic)
        x = x + self.ffwd(self.ln2(x), deterministic=deterministic)
        return x

def test_transformer_block():
    rng = jax.random.PRNGKey(0)
    B, S, D, H = 16, 10, 768, 12
    block = Block(hidden_dim=D, head_size=H, num_heads=D//H, dropout_rate=0.1)
    x = jax.random.normal(rng, (B, S, D))
    params = block.init(rng, x)
    out = block.apply(params, x)
    assert out.shape == (B, S, D)

class Transformer(nn.Module):
    hidden_dim: int = 768
    head_size: int = 12
    num_heads: int = 64
    num_layers: int = 12
    dropout_rate: float = 0.1

    def setup(self):
        self.blocks = [Block(hidden_dim=self.hidden_dim, head_size=self.head_size, num_heads=self.num_heads, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)]

    def __call__(self, x, deterministic: bool = True):
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x

def test_transformer():
    rng = jax.random.PRNGKey(0)
    B, S, D, H = 16, 10, 768, 12
    transformer = Transformer(hidden_dim=D, head_size=H, num_heads=D//H, num_layers=12, dropout_rate=0.1)
    x = jax.random.normal(rng, (B, S, D))
    params = transformer.init(rng, x)
    out = transformer.apply(params, x)
    # print(summary(transformer))
    assert out.shape == (B, S, D)

if __name__ == "__main__":
    print("Running tests...")
    test_attention_head()
    print("AttentionHead test passed")
    test_multi_head_attention()
    print("MultiHeadAttention test passed")
    test_transformer_block()
    print("Block test passed")
    test_transformer()
    print("Transformer test passed")