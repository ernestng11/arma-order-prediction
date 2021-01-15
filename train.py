import torch
import time
import datetime
import os
import numpy as np

import configs
configs.init()


class RollBack:
    '''
    Early stops the training if training loss doesn't improve after a given patience and rollback to the 
    previous best model
    '''

    def __init__(self, patience=7):
        '''
        Input: patience (int): How long to wait after last time training loss improved
        '''
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.roll_back = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.roll_back = True  # This results in load_checkpoint()

        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    def reset(self):
        '''
        Result rollback object once self.roll_back = True (load_checkpoint() is called)
        '''
        self.counter = 0
        self.roll_back = False


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def data_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_seconds = int(elapsed_time)
    return elapsed_seconds


class TrainModel:
    def __init__(self, model, data_generator, PATH):
        self.model = model
        self.datagen = data_generator
        self.n_classes = self.datagen.armax - self.datagen.armin + 1
        self.wmtrx = torch.ones(self.n_classes)
        self.PATH = PATH

    def logFile(self, epoch_count, rb_counter, rb_patience, accuracy, train_loss):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        log_text = f'DT:{current_time},Epoch:{epoch_count},Rollback_counter:{rb_counter}/{rb_patience},Accuracy:{accuracy:.2f},Train_loss:{train_loss:.3f} \n\n'

        f = open(self.PATH+"logfiles.txt", "a+")
        f.write(log_text)
        f.close()

    def initOptim(self, name, lr):
        '''
        Takes in 'adam', 'nagsmall', 'nag075' and 'nag095'
        '''
        if name == 'adam':  # beta 1 and 2 uses default values
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr)
        if name == 'nagsmall':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.01, nesterov=True)
        elif name == 'nag075':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.75, nesterov=True)
        elif name == 'nag095':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.95, nesterov=True)

    def initCriterion(self, weight=None):
        if weight is None:
            weight = self.wmtrx
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)

    def getParamCount(self):
        '''
        Returns number of trainable parameters in the model
        '''
        count = sum(p.numel()
                    for p in self.model.parameters() if p.requires_grad)
        print(f'The model has {count:,} trainable parameters')

    def prepareData(self):
        '''
        Might want to put this in the generate function of Dataset class instead
        '''
        data_dict = self.datagen.generate()
        data_size = data_dict['input'].size()[0]

        train_data = []

        for i in range(data_size):
            train_data.append(
                [data_dict['input'][i], data_dict['output'][i].squeeze(0)])

        dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=data_size, shuffle=True)

        return dataloader

    def train(self, dataloader):
        '''
        Inputs: data - Dictionary with keys 'input' and 'output'
                data['input'] - tensor of shape (batch_size, 1, 1000)
                data['output'] - tensor of shape (batch_size, 1)

        Enumerate through data and input each tensor with shape (1,1000) into the model
        '''
        local_mtrx = self.initErrMatrix()

        epoch_loss = 0.0
        for data in dataloader:
            inputs, labels = data[0].to(
                configs.device), data[1].to(configs.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

            for x, y in zip(labels, torch.max(outputs, 1).indices):
                local_mtrx[x.detach()][y.detach()] += 1
            # print(labels)
            # print(torch.max(outputs,1).indices)
            # Get accuracy for current batch
            acc_tensor = (labels == torch.max(
                outputs, 1).indices).float().detach().numpy()
            accuracy = float(100*acc_tensor.sum()/len(acc_tensor))

        return epoch_loss, local_mtrx, accuracy

    def initErrMatrix(self):
        '''
        Initialize a matrix to store ground truth and predicted labels
        '''
        n_classes = self.datagen.armax - self.datagen.armin + 1
        mtrx = torch.zeros(n_classes, n_classes, dtype=torch.long)
        return mtrx

    def updateCEweights(self, error_matrix):
        '''
        Function to update criterion weights
        '''
        prob_mtrx = torch.true_divide(
            error_matrix, torch.sum(error_matrix, dim=1))
        experrmtrx = torch.exp(prob_mtrx)
        expsumrow = torch.sum(experrmtrx, dim=1)
        CEMrowarr = -1*torch.diag(prob_mtrx) + torch.log(expsumrow)

        self.wmtrx = (self.wmtrx + CEMrowarr) * 0.5

        self.initCriterion(weight=self.wmtrx)

    def train_eval(self, max_epochs=5, numBatchesperEpoch=3, patience=1, optim='adam', lr=0.001, isCEWeightsDynamic=True):

        self.initOptim(optim, lr)
        self.initCriterion()
        rollback = RollBack(patience=patience)

        self.model.train()

        best_train_loss = float('inf')
        train_loss_list = []
        global_step_list = []

        # Initialize global error matrix
        global_mtrx = self.initErrMatrix()

        for epoch in range(max_epochs):
            start_time = time.time()
            train_loss = 0.0
            total_data_time = 0
            accuracy = 0
            for _ in range(numBatchesperEpoch):
                # Get time taken to generate time series data
                data_start = time.time()
                data = self.prepareData()
                data_end = time.time()
                total_data_time += data_time(data_start, data_end)

                # Training step
                loss_val, mtrx, acc = self.train(data)

                train_loss += loss_val/numBatchesperEpoch
                accuracy += acc/numBatchesperEpoch
                global_mtrx += mtrx

            # Update criterion weights
            if isCEWeightsDynamic:
                self.updateCEweights(global_mtrx)
            # Re-initialise global error matrix
            global_mtrx = self.initErrMatrix()

            train_loss_list.append(train_loss)
            global_step_list.append(epoch)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Time to generate data: {total_data_time}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tAccuracy: {accuracy:.2f} %')

            rollback(train_loss, self.model)

            self.logFile(epoch+1, rollback.counter,
                         patience, accuracy, train_loss)

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.save_checkpoint(epoch+1, train_loss, accuracy)

            # Roll back call
            if rollback.roll_back:
                best_train_loss = self.load_checkpoint()
                rollback.reset()
                continue

        history = {'errmtrx': global_mtrx, 'training_loss': train_loss_list,
                   'global_steps': global_step_list}

        return history

    def save_checkpoint(self, epoch_count, loss, accuracy):
        '''
        Save model checkpoint for resuming training
        PATH should end with '/'
        '''
        # Delete previous saved model
        for file in os.listdir(self.PATH):
            if file.split('.')[-1] == 'pt':
                os.remove(self.PATH+file)
                print('--Previous best model deleted--')

        # Model naming
        pid = str(os.getpid())
        current_time = datetime.datetime.now()
        model_name = str(epoch_count) + '_' + \
            current_time.strftime("%Y-%m-%d-%H%M%S") + '_' + pid

        model_configs = {
            'epoch': epoch_count,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        torch.save(model_configs, self.PATH+model_name+'.pt')

        print('--Current best model saved--')

    def load_checkpoint(self):
        '''
        Load previous best model for training
        '''
        # Find previous saved model
        for file in os.listdir(self.PATH):
            if file.split('.')[-1] == 'pt':
                file_path = self.PATH+file

        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        self.model.train()
        print('--Previous best model loaded--')
        return loss
