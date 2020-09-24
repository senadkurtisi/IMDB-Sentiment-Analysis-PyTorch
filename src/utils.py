import torch
from globals import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def get_dataloaders(train_set, valid_set, test_set):
    ''' Created dataloaders for train test and validation
        datasets.
    
    Arguments:
        train_set: train set
        valid_set: valid set
        test_set: test set
    Returns:
        train_loader(DataLoader): train dataloader
        valid_loader(DataLoader): valid dataloader
        test_loader(DataLoader): test dataloader
    '''
    train_loader = DataLoader(train_set, batch_size=net_config.batch_size, 
                                                        collate_fn=pad_trim)
    valid_loader = DataLoader(valid_set, batch_size=net_config.batch_size, 
                                                        collate_fn=pad_trim)
    test_loader = DataLoader(test_set, batch_size=net_config.batch_size, 
                                                        collate_fn=pad_trim)

    return train_loader, valid_loader, test_loader


def split_train_val(train_set):
    ''' Splits the given set into train and
        validation sets WRT split ratio
    
    Arguments:
        train_set: set to split
    Returns:
        train_set: train dataset
        valid_set: validation dataset
    '''
    train_num = int(SPLIT_RATIO*len(train_set))
    valid_num = len(train_set) - train_num

    generator = torch.Generator().manual_seed(SEED)
    train_set, valid_set = random_split(train_set, lengths=[train_num, valid_num],
                                        generator=generator)
    
    return train_set, valid_set


def pad_trim(data):
    ''' Pads or trims the batch of input data.
        
    Arguments:
        data (torch.Tensor): input batch
    Returns:
        new_input (torch.Tensor): padded/trimmed input
        labels (torch.Tensor): batch of output target labels
    '''
    data = list(zip(*data))
    # Extract target output labels
    labels = torch.tensor(data[0]).float().to(device)
    # Extract input data
    inputs = data[1]

    # Extract only the part of the input up to the MAX_SEQ_LEN point
    # if input sample contains more than MAX_SEQ_LEN. If not then
    # select entire sample and append <pad_id> until the length of the
    # sequence is MAX_SEQ_LEN
    new_input = torch.stack([torch.cat((input[:MAX_SEQ_LEN], 
                            torch.tensor([config['pad_id']]*max(0, MAX_SEQ_LEN - len(input))).long()))
                            for input in inputs])
    
    return new_input, labels


def calculate_acc(output, target):
    ''' Calculates binary accuracy based
        on given predictions and target labels.
    
    Arguments:
        output (torch.tensor): predictions
        target (torch.tensor): target labels
    Returns:
        acc (float): binary accuracy
    '''
    output = torch.round(output)
    correct = torch.sum(output==target).float()
    acc = (correct/len(target)).item()
    return acc