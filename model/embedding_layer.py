import torch
import torch.nn as nn

from model.char_cnn import CharCNN


class NERInputEmbedding(nn.Module):
    def __init__(self, vocab_size, word_dim, num_chars, char_dim, char_filters,
                 char_kernel, num_langs, lang_dim, pretrained_weights=None, dropout=0.5):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.word_embedding.weight.data.copy_(pretrained_weights)

        self.char_cnn = CharCNN(num_chars, char_dim, char_filters, char_kernel)
        self.lang_embedding = nn.Embedding(num_langs, lang_dim)
        self.drop = nn.Dropout(dropout)

        self.output_dim = word_dim + char_filters + lang_dim

    def forward(self, word_ids, char_ids, lang_ids):
        word_emb = self.word_embedding(word_ids)
        char_emb = self.char_cnn(char_ids)

        lang_emb = self.lang_embedding(lang_ids)
        lang_emb = lang_emb.unsqueeze(1).expand(-1, word_ids.size(1), -1)

        combined = torch.cat([word_emb, char_emb, lang_emb], dim=-1)
        return self.drop(combined)
