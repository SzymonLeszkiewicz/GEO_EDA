"""
Trajectory Prediction Model.

This module contains the implementation of a base model for trajectory prediction using an LSTM
(Long Short-Term Memory) architecture.
"""

import torch
import torch.nn as nn


class TrajectoryPredictionBaseModel(nn.Module):  # type: ignore
    """
    LSTM-based model for trajectory prediction.

    This model predicts future trajectories based on input sequences using an LSTM architecture.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int = 1,
            dropout_rate: float = 0.5,
    ):
        """
        Initializes the LSTM-based trajectory prediction model.

        Args:
            input_size (int): The number of input features for each time step.
            hidden_size (int): The number of features in the hidden state of
            the LSTM.
            output_size (int): The number of output features (e.g., the number
            of predicted trajectory points).
            num_layers (int, optional): The number of recurrent layers in the
            LSTM. Defaults to 1.
            dropout_rate (float, optional): The dropout rate for regularization.
            Defaults to 0.5.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size,
            sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size),
            representing the predicted future trajectories.
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Apply dropout to the output of the last LSTM cell
        out = self.dropout(out[:, -1, :])

        # Pass the output through the fully connected layer
        out = self.fc(out)

        return out
