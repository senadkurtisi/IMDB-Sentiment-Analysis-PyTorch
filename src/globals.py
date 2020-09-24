import torch
import torch.nn as nn
from argparse import ArgumentParser

# device on which we port the neural net and dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VOCAB HYPERPARAMETERS
# Maximum number of words in teh vocabulary
MAX_VOCAB_SIZE = 25000
# We include only words which occur in the corpus with
# some minimal frequency
MIN_FREQ = 10
#We trim/pad each sentence to this number of words
MAX_SEQ_LEN = 500

# DATASET HYPERPARAMETERS
# Split ratio between train and validation sed
SPLIT_RATIO = 0.8
# Random seed used for random train/validation split
SEED = 0

# Loss function <-> Default: Binary CrossEntropy
LOSS_FUNC = nn.BCELoss()

parser = ArgumentParser()
parser.add_argument("--pretrained_loc", type=str,
                    default="pretrained/imdb_model.pt",
                    help="Location of the pretrained parameters")
parser.add_argument("--mode", type=str, choices=["train", "test"], 
                    default="train", 
                    help="Mode in which we are using the neural net")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout probability")
parser.add_argument("--hidden_dim", type=int, default=100,
                    help="Dimension of the hidden/cell states")
parser.add_argument("--bidirectional", type=bool, default=True,
                    help="Is the RNN bidirectional?")
parser.add_argument("--epochs", type=int, default=10,
                    help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="Learning rate")
parser.add_argument("--layers", type=int, default=2,
                    help="Number of recurrent layers")
parser.add_argument("--batch_size", type=int, default=128)

net_config = parser.parse_args()

config = dict()

