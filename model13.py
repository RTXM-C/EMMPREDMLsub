import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class model13(nn.Module):
    def __init__(self, params, input_size=640, num_classes=9, input_channels=72):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(params['dropout_rate'])
        lstm_output_size = input_channels * params['hidden_size'] * 2
        self.fc1 = nn.Linear(lstm_output_size, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(len(out), -1)
        out = self.dropout(out)
        x = self.fc1(out)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
