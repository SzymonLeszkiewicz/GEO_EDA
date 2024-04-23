"""
Base model of regression module.

This module contains definition of base regression model.
"""
import torch
import torch.nn as nn


class RegressionBaseModel(nn.Module):
    """Regression base module.

    Definition of Regression Module
    """
    def __init__(self, embeddings_size, linear_sizes=None, activation_function=None):
        """Initializaiton of regression module.

        Args:
            embeddings_size (_type_): size of input embedding
            linear_sizes (_type_, optional): sizes of linear layers inside module. \
                Defaults to None.
            activation_function (_type_, optional): activation function from torch.nn
        """
        super().__init__()
        if linear_sizes is None:
            linear_sizes = [500, 1000]
        if activation_function is None:
            activation_function = nn.ReLU()
        self.model = torch.nn.Sequential()
        previous_size = embeddings_size
        for cnt, size in enumerate(linear_sizes):
            self.model.add_module(f"linear_{cnt}", nn.Linear(previous_size, size))
            self.model.add_module(f"ReLU_{cnt}", activation_function)
            previous_size = size
            if cnt % 2:
                self.model.add_module(f"dropout_{cnt}", nn.Dropout(p=0.2))
        self.model.add_module("linear_final", nn.Linear(previous_size, 1))

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        Args:
            x torch.Tensor: Vector data

        Returns:
            _type_: target value
        """
        return self.model(x)
