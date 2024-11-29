import torch
from torch import nn
from typing import Optional, Tuple


class CTC(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ctc_layer = nn.Linear(d_model, vocab_size + 1)
    
    def forward(self, enc_out):
        return torch.log_softmax(
            self.ctc_layer(enc_out),
            dim=-1,
        ) # (B, L, vocab_size + 1)

    def inference(self, enc_out):
        return torch.softmax(
            self.ctc_layer(enc_out),
            dim=-1,
        ) # (B, L, vocab_size + 1)
