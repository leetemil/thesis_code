import warnings
from collections import namedtuple
from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch import jit
from torch.nn.utils import weight_norm

# Inspired by https://github.com/guillitte/pytorch-sentiment-neuron/blob/master/models.py
class mLSTMCellJIT(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.wmx = nn.Linear(input_size, hidden_size, bias = False)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias = False)
        self.wx =  nn.Linear(input_size, 4 * hidden_size, bias = False)
        self.wh =  nn.Linear(hidden_size, 4 * hidden_size, bias = True)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = state

        m = self.wmx(input) * self.wmh(hx)
        gates = self.wx(input) + self.wh(m)
        ingate, forgetgate, outgate, cellgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class mLSTMLayerJIT(jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = mLSTMCellJIT(input_size, hidden_size)

    @jit.script_method
    def forward(self, input, state = None, mask = None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_size = input.size(0)
        seq_len = input.size(1)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, 1, device = input.device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)

        if state is None:
            state = (
                torch.zeros(batch_size, self.hidden_size, device = input.device),
                torch.zeros(batch_size, self.hidden_size, device = input.device)
            )

        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_state = state
            hx, cx = self.cell(input[:, i, :], state)

            seq_mask = mask[:, i]
            hx = seq_mask * hx + (1 - seq_mask) * prev_state[0]
            cx = seq_mask * cx + (1 - seq_mask) * prev_state[1]
            state = (hx, cx)

            outputs += [hx]
        return torch.stack(outputs, dim = 1), state

class mLSTMJIT(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        layers = [mLSTMLayerJIT(input_size, hidden_size)] + [mLSTMLayerJIT(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)

    @jit.script_method
    def forward(self, input, states = None, mask = None):
        # type: (Tensor, Optional[List[Tuple[Tensor, Tensor]]], Optional[Tensor]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for layer in self.layers:
            state = states[i] if states is not None else None
            output, out_state = layer(output, state, mask)
            output_states.append(out_state)
            i += 1
        return output, output_states

class mLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.wmx = nn.Linear(input_size, hidden_size, bias = False)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias = False)
        self.wx =  nn.Linear(input_size, 4 * hidden_size, bias = False)
        self.wh =  nn.Linear(hidden_size, 4 * hidden_size, bias = True)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx = state

        m = self.wmx(input) * self.wmh(hx)
        gates = self.wx(input) + self.wh(m)
        ingate, forgetgate, outgate, cellgate = torch.chunk(gates, 4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        cellgate = torch.tanh(cellgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class mLSTMLayer(torch.nn.Module):
    __constants__ = ['hidden_size']

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = mLSTMCell(input_size, hidden_size)

    def forward(self, input, state = None, mask = None):
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        batch_size = input.size(0)
        seq_len = input.size(1)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, 1, device = input.device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)

        if state is None:
            state = (
                torch.zeros(batch_size, self.hidden_size, device = input.device),
                torch.zeros(batch_size, self.hidden_size, device = input.device)
            )

        outputs = []
        for i in range(seq_len):
            prev_state = state
            hx, cx = self.cell(input[:, i, :], state)

            seq_mask = mask[:, i]
            hx = seq_mask * hx + (1 - seq_mask) * prev_state[0]
            cx = seq_mask * cx + (1 - seq_mask) * prev_state[1]
            state = (hx, cx)

            outputs += [hx]
        return torch.stack(outputs, dim = 1), state

class mLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        layers = [mLSTMLayer(input_size, hidden_size)] + [mLSTMLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)

    def forward(self, input, states = None, mask = None):
        # type: (Tensor, Optional[List[Tuple[Tensor, Tensor]]], Optional[Tensor]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        output_states = []
        output = input

        for i, layer in enumerate(self.layers):
            state = states[i] if states is not None else None
            output, out_state = layer(output, state, mask)
            output_states.append(out_state)
        return output, output_states
