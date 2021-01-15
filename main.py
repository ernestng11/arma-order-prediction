import train
import ModelControl
import dataset

import torch
import time
import datetime
import os

import configs
configs.init()
# General specifications
# Model specifications
# Training specifications

# Dataset Initialisation
dataset = dataset.Dataset()

# Model Building
model = ModelControl.createNetwork(depth=configs.depth)

# Training
train_model = train.TrainModel(model, dataset, PATH=configs.PATH_TO_LOG)

history = train_model.train_eval(
    max_epochs=configs.max_epochs,
    numBatchesperEpoch=configs.numBatchesperEpoch,
    patience=configs.patience,
    optim=configs.optimizer,
    lr=configs.initialLR,
    isCEWeightsDynamic=configs.isCEWeightsDynamic)
