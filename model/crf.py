import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_trans = nn.Parameter(torch.empty(num_tags))
        self.end_trans = nn.Parameter(torch.empty(num_tags))
        self._reset_params()

    def _reset_params(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        log_z = self._forward_algorithm(emissions, mask)
        gold = self._score_sentence(emissions, tags, mask)
        return (log_z - gold).mean()

    def _forward_algorithm(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape

        alpha = self.start_trans + emissions[:, 0]

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            scores = alpha.unsqueeze(2) + trans + emit
            new_alpha = torch.logsumexp(scores, dim=1)

            active = mask[:, t].unsqueeze(1)
            alpha = new_alpha * active + alpha * (1 - active)

        alpha = alpha + self.end_trans
        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        b_idx = torch.arange(batch_size, device=tags.device)

        score = self.start_trans[tags[:, 0]]
        score = score + emissions[b_idx, 0, tags[:, 0]]

        for t in range(1, seq_len):
            cur = tags[:, t]
            prev = tags[:, t - 1]
            active = mask[:, t]

            score = score + (self.transitions[prev, cur] + emissions[b_idx, t, cur]) * active

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[b_idx, lengths]
        score = score + self.end_trans[last_tags]

        return score

    def decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape

        score = self.start_trans + emissions[:, 0]
        history = []

        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            candidates = score.unsqueeze(2) + trans + emit

            best_score, best_prev = candidates.max(dim=1)
            history.append(best_prev)

            active = mask[:, t].unsqueeze(1)
            score = best_score * active + score * (1 - active)

        score = score + self.end_trans

        seq_lens = mask.long().sum(dim=1)
        best_last = score.argmax(dim=1)

        result = []
        for b in range(batch_size):
            length = seq_lens[b].item()
            path = [best_last[b].item()]
            for t in range(length - 2, -1, -1):
                path.append(history[t][b, path[-1]].item())
            path.reverse()
            path += [0] * (seq_len - length)
            result.append(path)

        return torch.tensor(result, device=emissions.device, dtype=torch.long)
