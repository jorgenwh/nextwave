import torch

class SimpleWaveNet(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=1028, num_layers=2, output_size=1, batch_size=1):
        super(SimpleWaveNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        #input_size = x.shape[2]

        r, _ = self.lstm(x)
        r = r.reshape(batch_size, seq_length, self.hidden_size)
        r = self.linear(r)

        return r
