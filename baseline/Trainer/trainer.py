import copy
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CustomDataset


class Trainer:
    def __init__(
        self,
        model: nn.Module = None,
        args: dict = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        train_df: pd.DataFrame | gpd.GeoDataFrame = None,
        eval_df: pd.DataFrame | gpd.GeoDataFrame = None,
        test_df: pd.DataFrame | gpd.GeoDataFrame = None,
        optimizer: torch.optim.Optimizer = None,
        # lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        loss_fn = nn.L1Loss(),
    ):

        self.model = model
        self.args = args
        batch_size = self.args.get("batch_size", 32)
        input_cols, labels = self.args.get(
            "input_col", "vector_embeddings"
        ), self.args.get("labels", "labels")

        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
            self.eval_dataloader = eval_dataloader
        else:

            self.train_dataset = CustomDataset(
                train_df[input_cols].values, train_df[labels].values
            )
            self.eval_dataset = CustomDataset(
                eval_df[input_cols].values, eval_df[labels].values
            )

            # Create DataLoader objects
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True
            )
            self.eval_dataloader = DataLoader(
                self.eval_dataset, batch_size=batch_size, shuffle=False
            )
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        else:
            if test_df is not None:

                self.test_dataset = CustomDataset(
                    test_df[input_cols].values, test_df[labels].values
                )
                self.test_dataloader = torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=batch_size, shuffle=False
                )

        # self.lr_scheduler = lr_scheduler
        # if self.lr_scheduler is not None:

        self.optimizer = optimizer

        self.device = self.args.get("device", "cuda")
        self.epochs = self.args.get("epochs", 50)
        self.loss_fn = loss_fn
        self.best_weights = None

    def train(self):
        best_mse = np.inf  # init to infinity
        l1_loss_eval = []
        l1_loss_train = []

        for epoch in range(self.epochs):
            loss_list = []
            self.model.train()
            for i, data in tqdm(
                enumerate(self.train_dataloader),
                desc=f"Epoch: {epoch}",
                total=len(self.train_dataloader),
            ):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            print(
                f"Epoch [{epoch+1}/{self.epochs}], avg_loss: {np.mean(loss_list):.4f}"
            )
            l1_loss_train.append(np.mean(loss_list))
            eval_loss = self.evaluate()
            l1_loss_eval.append(eval_loss)
            if eval_loss < best_mse:
                best_mse = eval_loss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                print(f"Best model found at epoch {epoch}, loss: {best_mse:.4f}")
        self.model.load_state_dict(self.best_weights)
        return self.model, l1_loss_train, l1_loss_eval

    def evaluate(self):
        with torch.no_grad():
            eval_loss_list = []
            self.model.eval()
            for i, data in tqdm(
                enumerate(self.eval_dataloader),
                desc="Evaluation",
                total=len(self.eval_dataloader),
            ):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                eval_loss_list.append(float(loss.item()))

            eval_loss = np.mean(eval_loss_list)
            print(f"Eval loss: {eval_loss:.4f}")

        return eval_loss

    def calculate_metrics(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
