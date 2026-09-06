import torch
import torch.nn as nn


class GRUDeadReckoning(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)

        # Use the final timestep
        last_output = output[:, -1, :]

        return self.fc(last_output)