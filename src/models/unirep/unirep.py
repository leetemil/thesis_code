import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class UniRep(nn.Module):
    def __init__(self, num_tokens = 30, padding_idx = 0, embed_size = 10, hidden_size = 512, num_layers = 1):
        super().__init__()

        # Define parameters
        self.num_tokens = num_tokens
        self.padding_idx = padding_idx
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(self.num_tokens, self.embed_size, padding_idx = self.padding_idx)

        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)

        self.lin = nn.Linear(self.hidden_size, self.num_tokens)

    def run_rnn(self, xb):
        # Get length of each sequence by looking at pad values
        lengths = (xb != self.padding_idx).sum(dim = 1)

        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Pack padded sequence
        packed_seq = pack_padded_sequence(embedding, lengths, batch_first = True, enforce_sorted = False)
        packed_out, last = self.rnn(packed_seq)
        return pad_packed_sequence(packed_out, batch_first = True)[0], lengths, last

    def predict(self, xb, lengths):
        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Pack padded sequence
        packed_seq = pack_padded_sequence(embedding, lengths, batch_first = True, enforce_sorted = False)
        packed_out, _ = self.rnn(packed_seq)
        out, _ = pad_packed_sequence(packed_out, batch_first = True)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        log_likelihoods = nn.functional.log_softmax(linear, dim = 2)
        return log_likelihoods

    def forward(self, xb):
        # Get length of each sequence by looking at pad values
        lengths = (xb != self.padding_idx).sum(dim = 1)

        pred = self.predict(xb, lengths)

        # Calculate loss
        true = torch.zeros(xb.shape, dtype = torch.int64, device = xb.device) + self.padding_idx
        true[:, :-1] = xb[:, 1:]

        # Flatten the sequence dimension to compare each timestep in cross entropy loss
        loss = F.nll_loss(pred.permute(0, 2, 1), true, ignore_index = self.padding_idx, reduction = "none")

        # Mean over sequence length first
        loss = loss.sum(dim = 1).div(lengths).mean()

        metrics_dict = {}
        return loss, metrics_dict

    def protein_logp(self, xb):
        # Get length of each sequence by looking at pad values
        lengths = (xb != self.padding_idx).sum(dim = 1)
        pred = self.predict(xb, lengths)

        # Calculate loss
        true = torch.zeros(xb.shape, dtype = torch.int64, device = xb.device) + self.padding_idx
        true[:, :-1] = xb[:, 1:]

        # Flatten the sequence dimension to compare each timestep in cross entropy loss
        loss = F.nll_loss(pred.permute(0, 2, 1), true, ignore_index = self.padding_idx, reduction = "none")
        log_probabilities = -1 * loss.sum(dim = 1)
        return log_probabilities

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"UniRep summary:\n"
                f"  RNN type:    {type(self.rnn).__name__}\n"
                f"  Embed size:  {self.embed_size}\n"
                f"  Hidden size: {self.hidden_size}\n"
                f"  Layers:      {self.num_layers}\n"
                f"  Parameters:  {num_params:,}\n")

    def save(self, f):
        args_dict = {
            "num_tokens": self.num_tokens,
            "padding_idx": self.padding_idx,
            "embed_size": self.embed_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers
        }

        torch.save({
            "name": "UniRep",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, f)
