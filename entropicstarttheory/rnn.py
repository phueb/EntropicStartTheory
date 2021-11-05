import torch
import numpy as np
from typing import Dict, Any


class RNN(torch.nn.Module):
    def __init__(self,
                 flavor: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool,
                 embeddings: np.array,
                 ):

        super().__init__()
        self.flavor = flavor
        self.input_size = input_size
        self.hidden_size = hidden_size

        # define operations
        self.embed = torch.nn.Embedding(input_size, hidden_size)  # embed_size does not have to be hidden_size
        if flavor == 'lstm':
            cell = torch.nn.LSTM
        elif flavor == 'srn':
            cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "flavor".')
        self.encode = cell(input_size=hidden_size,  # this does not have to be hidden_size
                           hidden_size=hidden_size,
                           batch_first=True,
                           bias=bias,
                           num_layers=num_layers,
                           dropout=0)
        self.project = torch.nn.Linear(in_features=hidden_size,
                                       out_features=input_size,
                                       bias=bias)

        # init weights - this is required to get good balanced accuracy
        max_w = np.sqrt(1 / hidden_size)

        self.project.weight.data.uniform_(-max_w, max_w)
        if bias:
            self.project.bias.data.fill_(0.0)
        # init embeddings (possible with pre-trained vectors)
        self.embed.weight.data = torch.from_numpy(embeddings.astype(np.float32))

        self.cuda()

        print(f'Initialized {flavor} with input_size={input_size}')

    def forward(self,
                inputs: torch.LongTensor
                ) -> Dict[str, Any]:

        embedded = self.embed(inputs)  # [batch_size, context_size, hidden_size]
        output_at_all_steps, hidden_at_last_step = self.encode(embedded)
        # note: when num_layers=1, output and hidden state are identical
        # state_at_all_steps has shape [batch_size, context_size, hidden_size]
        last_output = torch.squeeze(output_at_all_steps[:, -1])  # [batch_size, hidden_size]
        logits = self.project(last_output)  # [batch_size, input_size]

        res = {'last_output': last_output, 'logits': logits}

        if self.flavor == 'lstm':
            h_n, c_n = hidden_at_last_step
            res['h_n'] = h_n
            res['c_n'] = c_n
        elif self.flavor == 'rnn':
            res['h_n'] = hidden_at_last_step

        return res
