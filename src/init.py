import torch
import torchtext
import spacy
from torchtext.data import get_tokenizer
from torch.utils.data import random_split
from torchtext.experimental.datasets import IMDB

from globals import *
from utils import split_train_val

def init(config):
    ''' Loads the GloVe embeddings for the words
        which occur in the IMDB train set vocab 
        and uses that vocab to create train, validation
        and test sets for the IMDB dataset. Extracts the
        pad_id token.
    '''
    import os
    if not os.path.isdir('.data'):
        os.mkdir('.data')

    # Extract the initial vocab from the IMDB dataset
    vocab = IMDB(data_select='train')[0].get_vocab()
    # Create GloVe embeddings based on original vocab 
    # word freqs
    glove_vocab = torchtext.vocab.Vocab(counter=vocab.freqs, 
                                        max_size=MAX_VOCAB_SIZE, 
                                        min_freq=MIN_FREQ,
                                        vectors=torchtext.vocab.GloVe(name='6B'))

    # Acquire 'Spacy' tokenizer for the vocab words
    tokenizer = get_tokenizer('spacy')
    # Acquire train and test IMDB sets with previously created
    # GloVe vocab and 'Spacy' tokenizer 
    train_set, test_set = IMDB(tokenizer=tokenizer, vocab=glove_vocab)

    # Extract the vocab of the acquired train set
    vocab = train_set.get_vocab()
    # Extract the token used for padding
    pad_id = vocab['<pad>']

    # Split the train set into train and validation sets
    train_set, valid_set = split_train_val(train_set)

    config['train'] = train_set
    config['val'] = valid_set
    config['test'] = test_set
    config['vocab'] = vocab
    config['pad_id'] = pad_id

