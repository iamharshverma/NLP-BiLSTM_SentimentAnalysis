import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.num_layers = config['n_enc_layers']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.hidden_dim = config['hidden_dim']
        self.n_classes = config['n_classes']

        self.bidirectional = True
        self.batch_size = 5

        # For unidirectional LSTM Model
        # self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, self.num_layers,
        #                         bidirectional=True, dropout=self.dpout_model)

        # For Bi-idirectional LSTM Model
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)
        self.hidden2label = nn.Linear(self.hidden_dim, self.n_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))


    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]

        return emb

    # Changes in forward method For unidirectional LSTM Model
    # def forward(self, sent_tuple):
    #     # sent_len [max_len, ..., min_len] (batch)
    #     # sent (seqlen x batch x worddim)
    #
    #     sent, sent_len = sent_tuple
    #
    #     # Sort by length (keep idx)
    #     sent_len, idx_sort = np.ascontiguousarray(np.sort(sent_len)[::-1]), np.argsort(-sent_len)
    #     sent = sent.index_select(1, torch.LongTensor(idx_sort))
    #
    #     # Handling padding in Recurrent Networks
    #     sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
    #     sent_output = self.enc_lstm(sent_packed)[1][0].squeeze(0)  # batch x 2*nhid
    #
    #     # Un-sort by length
    #     idx_unsort = np.argsort(idx_sort)
    #     print(sent_output)
    #     emb = sent_output.index_select(0, torch.LongTensor(idx_unsort))
    #
    #     return emb

class NLINet(nn.Module):
    def __init__(self, config):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']
        self.encoder = eval(self.encoder_type)(config)

        # Initial Code uncomment for
        # self.inputdim = self.enc_lstm_dim
        # self.classifier = nn.Sequential(
        #     nn.Linear( self.inputdim, self.fc_dim),
        #     nn.Linear(self.fc_dim, self.fc_dim),
        #     nn.Linear(self.fc_dim, self.n_classes)
        #     )

        ## Handling input feature dimentions for bi-directional LSTM
        self.inputdim = 2 * self.enc_lstm_dim if self.encoder_type == "LSTMEncoder" else self.enc_lstm_dim

        # Adding handle for Non-Linear and Linear Classification
        # If non liner parameter is set then add dropout layers else just keep linear layers
        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Linear(self.fc_dim, self.n_classes)
            )

    def forward(self, s1):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        output = self.classifier(u)
        return output