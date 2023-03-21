from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dain import DAIN_Layer
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
import torch.optim as optim
import torch

class TempDataset(Dataset):
    def __init__(self, train_dl_norm, train_dl_target):
        super(Dataset, self).__init__()
        self.train_dl_norm = train_dl_norm
        self.train_dl_target = train_dl_target

    def __len__(self):
        return self.train_dl_norm.shape[0]

    def __getitem__(self, idx):
        x = self.train_dl_norm[idx,:,:]
        y = self.train_dl_target[idx,]
        return x, y

class MLP(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.0001):
        super(MLP, self).__init__()
        self.num_sensors = 10  # this is the number of features
        self.hidden_units = 5
        self.num_layers = 1
        self.base = nn.LSTM(
            input_size=10,
            hidden_size=5,
            batch_first=True,
            num_layers=1
        )

        self.dean = DAIN_Layer(mode=mode, mean_lr=mean_lr, gate_lr=gate_lr, scale_lr=scale_lr, input_dim=1)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dean(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.base(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def lob_epoch_trainer(model, loader, lr=0.0001, optimizer=optim.RMSprop):
    model.train()

    model_optimizer = optimizer([
        {'params': model.dean.mean_layer.parameters(), 'lr': lr * model.dean.mean_lr},
        {'params': model.dean.scaling_layer.parameters(), 'lr': lr * model.dean.scale_lr},
        {'params': model.dean.gating_layer.parameters(), 'lr': lr * model.dean.gate_lr},
    ], lr=lr)

    criterion = MSELoss()
    train_loss, counter = 0, 0

    for inputs, targets in loader:
        model_optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        # targets = torch.squeeze(targets)

        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        train_loss += loss.item()
        counter += inputs.size(0)

    loss = (loss / counter).cpu().data.numpy()
    return loss

def train_evaluate_anchored(model, loader, lr=0.00001, optimizer=optim.RMSprop, train_epochs=5):
    results = []
    model_optimizer = optimizer([
        {'params': model.dean.mean_layer.parameters(), 'lr': lr * model.dean.mean_lr},
        {'params': model.dean.scaling_layer.parameters(), 'lr': lr * model.dean.scale_lr},
        {'params': model.dean.gating_layer.parameters(), 'lr': lr * model.dean.gate_lr},
    ], lr=lr)

    criterion = MSELoss()

    for inputs, targets in loader:
        train_loss, counter = 0, 0
        model.eval()
        norm = model(inputs)
        norm = torch.squeeze(norm)
        if len(results) == 0:
            results = norm
        else:
            results = torch.cat((results, norm))

        model.train()
        for epoch in range(train_epochs):
            model_optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, targets)

            loss.backward()
            model_optimizer.step()

            train_loss += loss.item()
            counter += inputs.size(0)

        loss = (loss / counter).cpu().data.numpy()
        print("Epoch ", epoch, "loss: ", loss)

    return results




