import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Load pre-trained MobileNetV2 model
        net = MobileNetV2(width_mult=width_mult)
        if pretrain:
            state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar', map_location='cpu')
            net.load_state_dict(state_dict_mobilenet)

        # Use the features part of MobileNetV2
        self.cnn = nn.Sequential(*list(net.features)[:19])

        # LSTM layer
        self.rnn = nn.LSTM(
            int(1280 * width_mult if width_mult > 1.0 else 1280),
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layer
        if self.bidirectional:
            self.lin = nn.Linear(2 * self.lstm_hidden, 9)
        else:
            self.lin = nn.Linear(self.lstm_hidden, 9)

        # Dropout layer
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        # Initialize the hidden state and cell state of LSTM
        num_directions = 2 if self.bidirectional else 1
        return (Variable(torch.zeros(num_directions * self.lstm_layers, batch_size, self.lstm_hidden)),
                Variable(torch.zeros(num_directions * self.lstm_layers, batch_size, self.lstm_hidden)))

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)  # Global average pooling

        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out



