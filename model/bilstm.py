import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_dim = hidden_size * 2
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        sorted_lengths, sort_idx = lengths.sort(descending=True)
        sorted_x = x[sort_idx]

        packed = pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        _, unsort_idx = sort_idx.sort()
        output = output[unsort_idx]

        return self.drop(output)
