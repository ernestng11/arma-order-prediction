import torch
'''
INFO:

a) User can adjust
  - to use GPU or CPU for training, and which GPU number to be used.
  - number of epochs, learing rate decay, early stopping i.e consecutive number of training epochs without improvements.
  - whether to adjust the weights of cross entropy criterion dynamically during training.
  - the type of architecture to be trained. See parameter 'modelType'.
  - training for AR or MA predictions. see parameter 'isARmodel'.
  - optimizer to be used. See parameter 'optimType'.
  - range of hyperparameters depth, kW and features.
b) Use ModelControl to create different nn models of different hyperparameters
c) Train the models and records the correct percentages and mean errors.

'''

'''
Adjust parameters here
1. max_epochs
2. patience
3. initial LR
4. lr decay
5. numBatchesPerEpoch
6. isCEWeightsDynamic
'''


def init():
    # General specifications
    # Model specifications
    global nInputPlane
    global features
    global categories
    global kW_first
    global kW_second
    global dW_first
    global dW_second

    nInputPlane = 1
    categories = 10  # Change to dynamic assignment
    features = 300
    kW_first = 10
    kW_second = 1
    dW_first = 10
    dW_second = 1

    # Training specifications
    global PATH_TO_LOG
    global depth
    global max_epochs
    global numBatchesperEpoch
    global patience
    global initialLR
    global isCEWeightsDynamic
    global optimizer
    global device

    PATH_TO_LOG = './checkpoints/'
    depth = 10
    max_epochs = 50
    numBatchesperEpoch = 5
    patience = 5
    initialLR = 0.001
    isCEWeightsDynamic = True
    optimizer = 'adam'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
