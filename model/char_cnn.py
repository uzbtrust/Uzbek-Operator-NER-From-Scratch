import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(self, num_chars, char_embed_dim, num_filters, kernel_size, padding_idx=0):
        super().__init__()
        self.char_embedding = nn.Embedding(num_chars, char_embed_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(char_embed_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.char_embedding.weight)
        self.char_embedding.weight.data[padding_idx].fill_(0)

    def forward(self, char_ids):
        batch_size, seq_len, word_len = char_ids.shape
        flat = char_ids.view(batch_size * seq_len, word_len)

        embedded = self.char_embedding(flat)
        embedded = embedded.transpose(1, 2)

        conv_out = self.activation(self.conv(embedded))
        pooled = conv_out.max(dim=2)[0]

        return pooled.view(batch_size, seq_len, -1)
