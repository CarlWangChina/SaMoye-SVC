# This is Multi-reference timbre encoder

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm
from module.attentions import MultiHeadAttention

# https://arxiv.org/pdf/2201.03864,
class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels=192,
        hidden_size=512,
        out_channels=192,
        kernel_size=5,
        n_heads=4,
        ge_layer=2,
    ):
        super(MRTE, self).__init__()
        self.cross_attention_f0 = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.ssl_conv = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.f0_conv = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.post_conv = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc, ssl_mask, f0, f0_mask, style_embed):
        ssl_enc = self.ssl_conv(ssl_enc * ssl_mask)
        f0_enc = self.f0_conv(f0 * f0_mask)
        attn_ssl_f0 = self.cross_attention_f0(ssl_enc, f0_enc, f0_mask.unsqueeze(2))

        x = (
            ssl_enc
            + attn_ssl_f0
            + style_embed
        )
        x = self.post_conv(x * ssl_mask)
        return x


if __name__ == "__main__":
    content_enc = torch.randn(3, 192, 100)
    content_mask = torch.ones(3, 1, 100)
    ref_mel = torch.randn(3, 128, 30)
    ref_mask = torch.ones(3, 1, 30)
    model = MRTE()
    out = model(content_enc, content_mask, ref_mel, ref_mask)
    print(out.shape)
