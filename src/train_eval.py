import torch
import torch.optim as optim
from globals import *
from utils import calculate_acc


def train(net, train_loader, optimizer, loss_func):
    ''' Performs one training epoch of LSTM.

    Arguments:
        net (nn.Module): RNN (currently LSTM)
        train_loader (DataLoader): load object for train data
        optimizer: optimizer object for net parameters
        loss_func: criterion function used for backprop
    Returns:
        epoch_loss (torch.float): mean loss value for all
                                batches
        epoch_acc (torch.float): mean acc value for all batches
    '''
    net.train()

    epoch_loss = 0
    epoch_acc = 0

    for input, labels in train_loader:
        input, labels = input.to(device), labels.to(device)

        optimizer.zero_grad()
        output = net(input).squeeze(1)
        loss = loss_func(output, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += calculate_acc(output, labels)

    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)

    return epoch_loss, epoch_acc


def evaluate(net, set_loader, loss_func):
    ''' Evaluates the performance of the RNN
        on the given set.

    Arguments:
        net (nn.Module): RNN (currently LSTM)
        set_loader (DataLoader): load object for val/test data
        loss_func: criterion function used for backprop
    Returns:
        eval_loss (torch.float): mean loss value for all
                                batches
        eval_acc (torch.float): mean acc value for all batches
    '''
    net.eval()
    eval_loss = 0
    eval_acc = 0

    for input, labels in set_loader:
        input, labels = input.to(device), labels.to(device)

        output = net(input).squeeze(1)

        loss = loss_func(output, labels)

        eval_loss += loss.item()
        eval_acc += calculate_acc(output, labels)

    eval_loss /= len(set_loader)
    eval_acc /= len(set_loader)

    return eval_loss, eval_acc


def train_loop(net, train_loader, valid_loader, test_loader):
    ''' Trains the model for specified number of epochs.
        Training process of the model can be tracked 
        by tracking train&validation acc/loss.
    
    Arguments:
        net (nn.Module): neural net to train
        train_loader (DataLoader): load object for train data
        valid_loader (DataLoader): load object for validation data
        test_loader (DataLoader): load object for test data
    '''
    # Create an optimizer for the network
    optimizer = optim.Adam(net.parameters(), lr=net_config.lr)

    for epoch in range(net_config.epochs):
        print(f'Epoch: {epoch+1:02}')
        # Train the net for a single epoch
        train_loss, train_acc = train(net, train_loader, 
                                            optimizer, LOSS_FUNC)

        # Evaluate the net performance on the validation dataset
        with torch.no_grad(): 
            valid_loss, valid_acc = evaluate(net, valid_loader, 
                                                        LOSS_FUNC)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')