from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class NN(object):

    def __init__(
        self, nn_sequential,
        lr=.001, momentum=.9, init_range=[-1,1],
        max_grad_norm=.5
        ):
        """
        Initialize simple sequential neural network.
        """
        self.model = nn_sequential
        # init layers
        for e in self.model:
            if type(e) == torch.nn.modules.linear.Linear:
                nn.init.uniform_(e.weight, init_range[0], init_range[1])

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.max_grad_norm = max_grad_norm

    def get_param_count(self):
        total = 0
        for param in self.model.parameters():
            total += param.numel()
        return total

    def training_loop(self, dataloader, devset=None, num_epochs=100):
        self.model.train()
        self.losses = {'train': [], 'dev': []}

        for epoch in range(num_epochs):
            for data in dataloader:
                # set gradients to zero
                self.optimizer.zero_grad()
                feature, target = data
                # run forward pass
                pred = self.model(feature)
                # compute loss and gradients
                loss = self.criterion(pred, target)
                loss.backward()
                # print undefined gradients
                for param in self.model.parameters():
                    if torch.isnan(param.grad).any():
                        print("NaN detected in gradients.")
                # clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                # update parameters
                self.optimizer.step()
            
            self.losses['train'].append(loss.item())
            # dev loss
            if devset:
                self.losses['dev'].append(self.calculate_loss(devset))

    def calculate_loss(self, dataset):
        self.model.eval()

        dev_loss = 0
        with torch.no_grad():
            for features, targets in dataset:
                preds = self.model(features) # forward pass
                loss = self.criterion(preds, targets)
                dev_loss += loss.item()

        self.model.train()
        return dev_loss / len(dataset)

    def plot_losses(self, x_start=10, ylim=None):
        plt.plot(self.losses['train'][x_start:])
        plt.plot(self.losses['dev'][x_start:])
        if ylim:
            plt.ylim([0,ylim])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def calculate_metrics(self, test_dataset, y_scaler=None, verbose=True):
        self.model.eval()

        target = test_dataset.tensors[1]
        with torch.no_grad():
            y_pred = self.model(test_dataset.tensors[0])

        if y_scaler:
            target = y_scaler.inverse_transform(target)
            y_pred = y_scaler.inverse_transform(y_pred)

        mae = mean_absolute_error(target, y_pred)
        mse = mean_squared_error(target, y_pred)
        r2 = r2_score(target, y_pred)
        
        self.model.train()
        
        if verbose:
            print(f'Mean Absolute Error: {mae}')
            print(f'Mean Squared Error: {mse}')
            print(f'R-squared: {r2}')

        return mae, mse, r2