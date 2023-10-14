from torch import nn
from torch.nn import Sequential, Conv2d, Conv1d, BatchNorm1d, BatchNorm2d, GRU, ReLU, Linear, LayerNorm

import torch

from hw_asr.base import BaseModel


class GRU_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0., batch_first=True):
        super().__init__()
        self.activation = ReLU()
        self.bn = BatchNorm1d(input_size)
        self.rnn_block = GRU(input_size, hidden_size, num_layers=1,
                             dropout=dropout, batch_first=batch_first)

    def forward(self, x):
        x = self.activation(self.bn(x))
        x = self.rnn_block(x.transpose(1, 2))[0]
        return x.transpose(1, 2)


class GRU_extended(nn.Module):
    """
    GRU RNN with optional batch normalization not implemented in Pytorchs' GRU :(((
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0., batch_first=True):
        super().__init__()
        self.rnn = []
        for _ in range(num_layers):
            self.rnn.append(
                GRU_block(input_size, hidden_size, dropout, batch_first))
            input_size = hidden_size
        self.rnn = Sequential(*self.rnn)

    def forward(self, x):
        return self.rnn(x)


class ConvLookahead(nn.Module):
    """
    Lookahead convolutional layer for DeepSpeech model.

    Keyword arguments:
    d -- dimention of sequence elements
    t -- context size
    """

    def __init__(self, d, t):
        super().__init__()
        self.t = t
        self.d = d
        self.conv = Conv1d(d, d, kernel_size=t+1, padding=t, groups=d)

    def forward(self, x):
        y = self.conv(x)[:, :, self.t:]
        return y.transpose(1, 2)


class DeepSpeechModel(BaseModel):
    """
    DeepSpeech model.

    Keyword arguments:
    n_convs, int -- number of 2D-convolutional layers
    conv_params, list of len n_convs -- parameters for every 2D-convolutional layer
    gru_params, dict -- parameters for GRU RNN
    lookahead_timesteps, int -- context size for lookahead layer after RNN
    """

    def __init__(self, n_feats, n_class, n_convs, conv_params, gru_params, lookahead_timesteps, **batch):
        super().__init__(n_feats, n_class, **batch)
        conv_list = []
        self.n_convs = n_convs
        self.conv_params = conv_params
        for i in range(n_convs):
            conv_list.append(Conv2d(**conv_params[i]))
            conv_list.append(BatchNorm2d(conv_params[i]["out_channels"]))
            conv_list.append(ReLU())
        self.conv_block = Sequential(*conv_list)
        self.gru = GRU_extended(
            self.transform_input_freq(n_feats), **gru_params)
        self.head = Sequential(
            ConvLookahead(gru_params["hidden_size"], lookahead_timesteps),
            LayerNorm(gru_params["hidden_size"]),
            Linear(gru_params["hidden_size"], n_class),
        )

    def forward(self, spectrogram, **batch):
        conv_out = self.conv_block(torch.unsqueeze(spectrogram, 1))
        conv_out = conv_out.view(conv_out.shape[0], -1, conv_out.shape[-1])
        gru_out = self.gru(conv_out)
        logits = self.head(gru_out)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for i in range(self.n_convs):
            filter, stride = self.conv_params[i]["kernel_size"][1], self.conv_params[i]["stride"][1]
            output_lengths = (output_lengths - filter) // stride + 1
        return output_lengths

    def transform_input_freq(self, input_freq):
        output_freq = input_freq
        for i in range(self.n_convs):
            filter, stride = self.conv_params[i]["kernel_size"][0], self.conv_params[i]["stride"][0]
            output_freq = (output_freq - filter) // stride + 1
        return output_freq * self.conv_params[-1]["out_channels"]
