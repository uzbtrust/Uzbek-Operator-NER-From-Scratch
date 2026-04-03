import torch
import torch.nn as nn

from model.embedding_layer import NERInputEmbedding
from model.bilstm import BiLSTMEncoder
from model.crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_chars, num_tags, word_dim=300, char_dim=50,
                 char_filters=50, char_kernel=3, num_langs=2, lang_dim=16,
                 hidden_size=256, num_layers=2, dropout=0.5, pretrained_weights=None):
        super().__init__()

        self.embedding = NERInputEmbedding(
            vocab_size=vocab_size,
            word_dim=word_dim,
            num_chars=num_chars,
            char_dim=char_dim,
            char_filters=char_filters,
            char_kernel=char_kernel,
            num_langs=num_langs,
            lang_dim=lang_dim,
            pretrained_weights=pretrained_weights,
            dropout=dropout,
        )

        self.encoder = BiLSTMEncoder(
            input_dim=self.embedding.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.projection = nn.Linear(self.encoder.output_dim, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, word_ids, char_ids, lang_ids, tags, mask, lengths):
        embedded = self.embedding(word_ids, char_ids, lang_ids)
        encoded = self.encoder(embedded, lengths)
        emissions = self.projection(encoded)
        loss = self.crf(emissions, tags, mask)
        return loss

    def predict(self, word_ids, char_ids, lang_ids, mask, lengths):
        embedded = self.embedding(word_ids, char_ids, lang_ids)
        encoded = self.encoder(embedded, lengths)
        emissions = self.projection(encoded)
        return self.crf.decode(emissions, mask)
