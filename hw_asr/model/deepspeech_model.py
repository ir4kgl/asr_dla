from torch import nn
from torch.nn import Sequential, Conv2d, Conv1d, BatchNorm2d, ReLU, GRU, Softmax, Linear

from torch import unsqueeze, squeeze, randn, mean

from hw_asr.base import BaseModel


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
        x_t = x.transpose(1, 2)
        y_t = self.conv(x_t)[:, :, self.t:]
        return y_t.transpose(1, 2)


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
        self.gru = GRU(**gru_params)
        self.head = Sequential(
            ConvLookahead(gru_params["hidden_size"], lookahead_timesteps),
            Linear(gru_params["hidden_size"], n_class),
        )

    def forward(self, spectrogram, **batch):
        spectrogram_t = unsqueeze(spectrogram.transpose(1, 2), 1)
        spectrogram_convolved = mean(self.conv_block(spectrogram_t), dim=1)
        h0 = randn(self.gru.num_layers, len(spectrogram), self.gru.hidden_size)
        gru_output, _ = self.gru(spectrogram_convolved, h0)
        logits = self.head(gru_output)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for i in range(self.n_convs):
            filter, stride = self.conv_params[i]["kernel_size"][0], self.conv_params[i]["stride"][0]
            output_lengths = (output_lengths - filter) // stride + 1
        return output_lengths
