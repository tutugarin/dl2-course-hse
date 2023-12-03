import math
import torch
from torch import nn

from dataset import Tokenizer, create_mask


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(1, max_len, self.embed_dim)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        layer_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_norm)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.apply(LanguageModel._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x, mask):
        x = self.pos_enc(self.embed(x))
        x = self.encoder(x, mask)
        output = self.head(x)
        return output

    def inference(self, tokenizer: Tokenizer, prompt, max_new_tokens=128, temperature=0.1, top_k=10):
        self.eval()
        device = next(self.parameters()).device

        tokenized_prompt = [tokenizer.bos_id] + tokenizer.encode(prompt)
        tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long, device=device)[None, ...]

        final_tokens = []
        context_tokens = tokenized_prompt
        for _ in range(max_new_tokens):
            mask, _ = create_mask(context_tokens)
            mask = mask.to(device)
            logits = self(context_tokens, mask)[:, -1, :]

            logits = logits / temperature
            v, _ = torch.topk(logits, top_k)
            threshold = v[0][-1]
            logits[logits < threshold] = 0
            next_token = torch.multinomial(logits, num_samples=1)

            context_tokens = torch.cat((context_tokens, next_token), dim=1)
            final_tokens.append(next_token.item())

            if next_token.item() == tokenizer.eos_id:
                break

        return prompt + tokenizer.decode(final_tokens)
