class CustomTransformerBlock(nn.Module):

    def __init__(self,config):
        super().__init__()
        ## nn.Linear can be used to represent a matrix
        # self.Q = nn.Linear(config.n_embd, config.n_embd)
        # self.K = nn.Linear(config.n_embd, config.n_embd)
        # self.V = nn.Linear(config.n_embd, config.n_embd)
        # But I want the raw Parameter
        # we sample from 1/embedding_dimension
        self.Q = nn.Parameter(torch.randn(config.n_embd, config.n_embd)*torch.sqrt(torch.tensor(1.0/config.n_embd)))
        self.K = nn.Parameter(torch.randn(config.n_embd, config.n_embd)*torch.sqrt(torch.tensor(1.0/config.n_embd)))
        self.V = nn.Parameter(torch.randn(config.n_embd, config.n_embd)*torch.sqrt(torch.tensor(1.0/config.n_embd)))

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
        )
    

    def attention(X, Q, K, V, mask=None, dropout=None):
        """
        X = incoming batch of tokens of dimension (batch_size, seq_len, n_embd)

        attention for a incoming 
        """ 

        key_t = torch.transpose(key, 0, 1)

        # queries : (batch_size, seq_len, n_embd) x (n_embd, n_embd) = (batch_size, seq_len, n_embd)
        queries = torch.matmul(X, Q)
        # keys : (batch_size, seq_len, n_embd) x (n_embd, n_embd) = (batch_size, seq_len, n_embd)
        keys = torch.matmul(X, K)
        # values : (batch_size, seq_len, n_embd) x (n_embd, n_embd) = (batch_size, seq_len, n_embd)
        values = torch.matmul(X, V)

        keys_transpose = torch.transpose(keys, 0, 1)

        # q_k_t : (batch_size, seq_len, n_embd) x (n_embd, n_embd) = (batch_size, seq_len, seq_len)
        q_k_t = torch.matmul(queries, keys_transpose)
        torch.matmul(torch.matmul(queries, keys_transpose)*torch.sqrt(torch.tensor(1.0/config.n_embd))
