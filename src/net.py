import torch
import torch.nn as nn

class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab, hidden_dim, layers, dropout=0.5, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim, num_layers=layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2*hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input):
        # Get the word embeddings of the batch
        embedded = self.embedding(input)
        # Propagate the input through LSTM layer/s
        _, (hidden, _) = self.lstm(embedded)

        # Extract output of the last time step
        # Extract forward and backward hidden states of the
        # last time step
        out = torch.cat([hidden[-2,:,:],hidden[-1,:,:]], dim=1)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.sigmoid(out)

        return out