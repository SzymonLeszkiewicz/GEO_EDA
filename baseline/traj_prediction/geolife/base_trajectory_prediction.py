"""
LSTM Model for travel time prediction.

This module defines an LSTM-based model for travel time prediction.
"""

import torch
import torch.nn as nn


class TrajectoryPredictionBaseModel(nn.Module):
    """LSTM-based model for travel time prediction.

    This model predicts travel time using Long Short-Term Memory (LSTM) architecture.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout_rate: float = 0.5,
    ):
        """Initialization of LSTM-based travel time prediction model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            output_size (int): Number of output features.
            num_layers (int, optional): The number of recurrent layers in the LSTM. Defaults to 1.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
        """
        super(TrajectoryPredictionBaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size),
            representing the predicted travel time.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
