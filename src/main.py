import torch
from globals import *

from init import init
from net import SentimentAnalyzer

from train_eval import train_loop, evaluate
from utils import get_dataloaders

if __name__ == "__main__":
    init(config)

    if net_config.mode == "train":
        net = SentimentAnalyzer(config['vocab'], net_config.hidden_dim,
                                net_config.layers, net_config.dropout, 
                                net_config.bidirectional).to(device)
        # Get dataloaders for train/validation/test sets
        train_loader, valid_loader, test_loader = get_dataloaders(config['train'], 
                                                                  config['val'], 
                                                                  config['test'])
        # Train the network
        train_loop(net, train_loader, valid_loader, test_loader)
    else:
        net = SentimentAnalyzer(config['vocab'], net_config.hidden_dim,
                                net_config.layers, net_config.dropout, 
                                net_config.bidirectional).to(device)
        # Load pretrained model parameters
        net.load_state_dict(torch.load(net_config.pretrained_loc))

    # Evaluate the network perfomance
    with torch.no_grad():
        test_loss, test_acc = evaluate(net, test_loader, LOSS_FUNC)

    print(f'Test. Loss: {test_loss:.3f} | Test. Acc: {test_acc*100:.2f}%')