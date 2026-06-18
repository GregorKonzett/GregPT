from torch import nn
import torch
from torch.functional import F

from tokenizer.TikTokenTokenizer import TikTokenTokenizer

dropout = 0.02
dropout1 = 0.02
embed_dim = 512
num_heads = 8
head_size = embed_dim // num_heads
block_count = 8
block_size = 1024
latent_dim = 128
rope_dim = 8
content_dim = head_size - rope_dim

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class MultiHeadLatentAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_down = nn.Linear(embed_dim, latent_dim, bias=False)
        self.vs_up = nn.Linear(latent_dim, embed_dim, bias=False)
        self.qs_content = nn.Linear(embed_dim, num_heads * content_dim, bias=False)
        self.qs_rope = nn.Linear(embed_dim, num_heads * rope_dim, bias=False)
        self.ks_content_up = nn.Linear(latent_dim, num_heads * content_dim, bias=False)
        self.ks_rope = nn.Linear(embed_dim, num_heads * rope_dim, bias=False)


        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def apply_rope(self, x, start_pos = 0):
        B, H, T, hs = x.shape

        pos = torch.arange(start_pos, T + start_pos, device=x.device)
        dim = torch.arange(0, hs, 2, device=x.device)

        inv_freq = 1.0 / (10000 ** (dim.float() / hs))
        freqs = torch.outer(pos.float(), inv_freq)  # (T, hs/2)

        cos = freqs.cos()[None, None, :, :]  # (1, 1, T, hs/2)
        sin = freqs.sin()[None, None, :, :]  # (1, 1, T, hs/2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.stack(
            [x_even * cos - x_odd * sin,
             x_even * sin + x_odd * cos],
            dim=-1
        )

        return x_rotated.flatten(-2)

    def forward(self, x, kv_cache=None, start_pos = 0, use_cache = False):
        B, T, C = x.shape

        down_kv = self.kv_down(x)  # [B, T, latent_dim]

        qs_content = self.qs_content(x).view(B, T, num_heads, content_dim).transpose(1, 2)
        qs_rope = self.qs_rope(x).view(B, T, num_heads, rope_dim).transpose(1, 2)
        ks_rope = self.ks_rope(x).view(B, T, num_heads, rope_dim).transpose(1, 2)

        qs_rope = self.apply_rope(qs_rope, start_pos)
        ks_rope = self.apply_rope(ks_rope, start_pos)

        if use_cache:
            if kv_cache is not None:
                old_down_kv, old_ks_rope = kv_cache
                down_kv = torch.cat([old_down_kv, down_kv], dim=1)
                ks_rope = torch.cat([old_ks_rope, ks_rope], dim=-2)

            if down_kv.shape[1] > block_size:
                down_kv = down_kv[:, -block_size:, :]
                ks_rope = ks_rope[:, :, -block_size:, :]

            new_cache = (down_kv, ks_rope)
        else:
            new_cache = None

        ks_content_weight = self.ks_content_up.weight.view(num_heads, content_dim, latent_dim)
        qs_latent = torch.einsum("bhtd,hdr->bhtr", qs_content, ks_content_weight)

        content_wei = torch.einsum("bhtr,bsr->bhts", qs_latent, down_kv)
        rope_wei = qs_rope @ ks_rope.transpose(-2, -1)
        wei = (content_wei + rope_wei) * (head_size ** -0.5)
        if not use_cache or T > 1:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        latent_out = torch.einsum("bhts,bsr->bhtr", wei, down_kv)

        vs_weight = self.vs_up.weight.view(num_heads, head_size, latent_dim)
        out = torch.einsum("bhtr,hdr->bhtd", latent_out, vs_weight)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout1(self.proj(out))
        return out, new_cache

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, idx):
        return self.net(idx)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_norm1 = nn.LayerNorm(embed_dim)
        self.ln_norm2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward()
        self.multihead_attn = MultiHeadLatentAttention()

    def forward(self, x, kv_cache=None, start_pos = 0, use_cache = False):
        tmp, new_cache = self.multihead_attn(self.ln_norm1(x), kv_cache, start_pos, use_cache)
        x = x + tmp
        x = x + self.ffwd(self.ln_norm2(x))
        return x, new_cache

class GptModel(nn.Module):
    def __init__(self, vocab_size, tokenizer: TikTokenTokenizer):
        super().__init__()
        self.device = get_device()
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.block_count = block_count
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embed_dim, device=self.device)
        self.blocks = nn.ModuleList([Block() for _ in range(self.block_count)])
        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.ln_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, targets=None, kv_caches = None, start_pos = 0, use_cache = False):
        B, T = x.shape

        new_caches = [] if use_cache else None

        x = self.token_embedding_table(x)

        for i, block in enumerate(self.blocks):
            if use_cache:
                block_cache = None if kv_caches is None else kv_caches[i]
                x, new_cache = block(x, block_cache, start_pos, use_cache)
                new_caches.append(new_cache)
            else:
                x, _ = block(x)

        x = self.ln_norm(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_caches

    def sample(self, logits):
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens = 512, temperature = 0.7):
        assert idx.shape[1] <= self.block_size
        self.eval()

        caches = None

        # Build cache first
        logits, _, caches = self(idx, kv_caches = caches, start_pos = 0, use_cache = True)
        logits = logits[:, -1, :] / temperature
        idx_next = self.sample(logits)
        next_token = idx_next[0, 0].item()
        prompt_len = idx.shape[1]
        idx = torch.cat([idx, idx_next], dim=1)

        abs_idx = prompt_len
        i = 1

        while i < max_new_tokens and not self.tokenizer.is_eos(next_token):
            logits, _, caches = self(idx_next, kv_caches = caches, start_pos = abs_idx, use_cache = True)
            logits = logits[:, -1, :] / temperature
            idx_next = self.sample(logits)
            next_token = idx_next[0, 0].item()
            idx = torch.cat([idx, idx_next], dim=1)

            abs_idx += 1
            i += 1

        self.train()

        return idx
