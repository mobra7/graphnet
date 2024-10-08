from abc import ABC, abstractmethod
from copy import deepcopy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,1,0"
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type,
                    Union, cast)
from tqdm import tqdm

import math
import numpy as np
import pandas as pd
import random
import sqlite3
from scipy.linalg import null_space
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR, GRAPHNET_ROOT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataset import Dataset, ParquetDataset, SQLiteDataset
from graphnet.data.dataset.dataset import ColumnMissingException, EnsembleDataset, parse_graph_definition
from graphnet.data.utilities.string_selection_resolver import (
    StringSelectionResolver
    )
from graphnet.models import Model, StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdge
from graphnet.models.gnn.gnn import GNN
from graphnet.models.graphs import GraphDefinition, KNNGraph
from graphnet.models.task import StandardLearnedTask
from graphnet.models.task.classification import freedom_BinaryClassificationTask
from graphnet.training.callbacks import PiecewiseLinearLR, ProgressBar

from graphnet.training.labels import Label
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.training.utils import collate_fn
from graphnet.utilities.config import Configurable, DatasetConfig,DatasetConfigSaverABCMeta, ModelConfig
from graphnet.utilities.logging import Logger

from freedom import (LikelihoodFreeModel,
                      make_train_validation_dataloader,
                      ScrambledDirection)

torch.multiprocessing.set_sharing_strategy('file_descriptor')

class disc_NeuralNetwork(GNN):
    def __init__(self, input_size, output_size, hidden_sizes=[150,250,400,400,250,150,100,64,32,8]):
        super().__init__(input_size,output_size)
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6 = nn.Linear(hidden_sizes[4], hidden_sizes[5])
        self.fc7 = nn.Linear(hidden_sizes[5], hidden_sizes[6])
        self.fc8 = nn.Linear(hidden_sizes[6], hidden_sizes[7])
        self.fc9 = nn.Linear(hidden_sizes[7], hidden_sizes[8])
        self.fc10 = nn.Linear(hidden_sizes[8], hidden_sizes[9])
        self.fc11 = nn.Linear(hidden_sizes[9], output_size)

    def forward(self, data):
        x = data
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        x = F.leaky_relu(self.fc7(x))
        x = F.leaky_relu(self.fc8(x))
        x = F.leaky_relu(self.fc9(x))
        x = F.leaky_relu(self.fc10(x))
        x = self.fc11(x)
        return x
    

class SNREModel(StandardModel):

    def __init__(
        self,
        *,
        discriminator: Union[torch.nn.Module, Model],
        scrambled_target: str,
        graph_definition: GraphDefinition,
        model_1 : Model = None,
        scramble_flag: str = 'scrambled_class',
        backbone: Model = None,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `SNREModel`."""
        # Just one task
        assert len(tasks) == 1

        # Only works with binary classification
        assert isinstance(tasks[0], freedom_BinaryClassificationTask)

        super().__init__(graph_definition = graph_definition,
                         backbone=backbone,
                         tasks = tasks,
                         optimizer_class = optimizer_class,
                         optimizer_kwargs = optimizer_kwargs,
                         scheduler_class = scheduler_class,
                         scheduler_kwargs = scheduler_kwargs,
                         scheduler_config = scheduler_config)
        
        self._model_1 = model_1
        for i,param in enumerate(self._model_1.parameters()):
            param.requires_grad = False
            
        self._discriminator = discriminator

        # grab name of scrambled target label e.g. direction
        self._scramble_flag = scramble_flag
        self._scrambled_target = scrambled_target

    def forward(self, data: Union[Data, List[Data]]) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]

        x1 = None
        x2_list = []
        y_scrambled_list = []
        self._model_1.inference()
        self._model_1.train(mode=False)

        for d in data:
            x = self._model_1.forward(d)[0]
            if x1 is None:
                x1 = x
            else:
                x1 = torch.cat((x1, x), dim=0)
            
            if not self._tasks[0]._inference:
                x = torch.clamp(x, min=1e-16)
                f = 1 / (1 + 1 / x)
                c = torch.where(d[self._scramble_flag] == 0, f, torch.zeros_like(f))
                d['snre_importance'] = torch.where(
                    d[self._scramble_flag] == 1,
                    torch.ones_like(f),
                    torch.sum((d[self._scramble_flag] == 0).float()) * f / torch.sum(c)
                )
                if torch.any(torch.isnan(f)) or torch.any(torch.isinf(f)):
                    print("f contains NaN or Inf values")
                if torch.any(torch.isnan(c)) or torch.any(torch.isinf(c)):
                    print("c contains NaN or Inf values")
            x = self.backbone(d)
            x2_list.append(x)
            y_scrambled_list.append(d[self._scrambled_target])
            

        x = torch.cat(x2_list, dim=0)
        y_scrambled = torch.cat(y_scrambled_list, dim=0)
        x = torch.cat([x, y_scrambled], dim=1)
        x2 = self._discriminator(x)

        if not self._tasks[0]._inference:
            preds = [task(x2) for task in self._tasks]
        else:
            preds = [task(x2) + x1 for task in self._tasks]

        return preds

    

    
def main(
    path: str,
    save_path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Run example."""

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/checkpoints/', exist_ok=True)
    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = f"{save_path}/wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="SNRE",
            entity="mobra-technical-university-of-munich",
            save_dir=wandb_dir,
            log_model=True,
        )

        # Constants
    features = FEATURES.ICECUBE86
    truth = TRUTH.ICECUBE86

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph representation
    graph_definition = KNNGraph(detector=IceCube86())

    # add your labels

    labels = {'scrambled_direction': ScrambledDirection(
        zenith_key='zenith',azimuth_key='azimuth'
        )
    }

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        db=config["path"],
        graph_definition=graph_definition,
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        truth_table=truth_table,
        labels= labels,
        selection= None, #either None, str, or List[(event_no,scramble_class)]
    )
    
    model_1 = Model.load('./vMF_IS_09_19/model.pth')

    pretrained_dynedge = Model.load('/scratch/users/mbranden/graphnet/playground/dynedge_baseline_3/model.pth')

    backbone = pretrained_dynedge.backbone
    for i,param in enumerate(backbone.parameters()):
        param.requires_grad = False
        if i == len(list(backbone.parameters())) - 3:
            break


    task = freedom_BinaryClassificationTask(
        hidden_size=1,
        target_labels=config["target"],
        loss_function= BinaryCrossEntropyLoss(),
        loss_weight = 'snre_importance'
    )
    discriminator = disc_NeuralNetwork(backbone.nb_outputs+3, 1)
    

    model_2 = SNREModel(
        graph_definition=graph_definition,
        backbone=backbone,
        discriminator=discriminator,
        model_1=model_1,
        scrambled_target="scrambled_direction",
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-3},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={'patience': 4, 'factor': 0.1},
        scheduler_config={'frequency': 1, 'monitor': 'val_loss'},
    )

    model_2.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        callbacks= [ProgressBar(),
                    EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    ),
                    ModelCheckpoint(
                        save_top_k=-1,
                        every_n_epochs = 5,
                        dirpath=f"{save_path}/checkpoints/",
                        filename=f"{model_2.backbone.__class__.__name__}"
                        + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}"
                    ),
                    ModelCheckpoint(
                        save_top_k=1,
                        monitor="val_loss",
                        mode="min",
                        dirpath=f"{save_path}/checkpoints/",
                        filename=f"best"
                        + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                    )],
        #ckpt_path = './SNRE_09_24/checkpoints/DynEdge-epoch=49-val_loss=0.01-train_loss=0.01.ckpt',
        **config["fit"]
    )

    model_2.save(f"{save_path}/model.pth")
    model_2.save_state_dict(f"{save_path}/state_dict.pth")
    model_2.save_config(f"{save_path}/model_config.yml")

if __name__ == "__main__":

    # settings
    path = "/scratch/users/allorana/northern_sqlite/files_no_hlc/dev_northern_tracks_full_part_1.db"
    save_path = '/scratch/users/mbranden/graphnet/playground/SNRE_10_02'

    pulsemap = 'InIcePulses'
    target = 'scrambled_class'
    truth_table = 'truth'
    gpus = [1]
    max_epochs = 250
    early_stopping_patience = 9
    batch_size = 500
    num_workers = 30
    wandb =  True
    

    main(
            path=path,
            save_path = save_path,
            pulsemap = pulsemap,
            target = target,
            truth_table = truth_table,
            gpus = gpus,
            max_epochs = max_epochs,
            early_stopping_patience = early_stopping_patience,
            batch_size = batch_size,
            num_workers = num_workers,
            wandb = wandb,
        )