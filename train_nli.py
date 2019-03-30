import os
import sys
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/stsa/', help="stsa data path ")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt",help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dpout_model", type=float, default=0.2, help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.3, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1.0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam", help="adam or sgd,lr=0.1")

# model
parser.add_argument("--encoder_type", type=str, default='LSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=128, help="encoder nhid dimension")
# Adding new Hidden Layer Dimentions Parameter
parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=128, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="positive/negative")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + valid['s1'] + test['s1'] , params.word_emb_path)

for split in ['s1']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
                                           [word for word in sent.split() if word in word_vec] +
                                           ['</s>'] for sent in eval(data_type)[split]])

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words': len(word_vec),
    'word_emb_dim': params.word_emb_dim,
    'enc_lstm_dim': params.enc_lstm_dim,
    'n_enc_layers': params.n_enc_layers,
    'dpout_model': params.dpout_model,
    'dpout_fc': params.dpout_fc,
    'fc_dim': params.fc_dim,
    'bsize': params.batch_size,
    'n_classes': params.n_classes,
    'pool_type': params.pool_type,
    'nonlinear_fc': params.nonlinear_fc,
    'encoder_type': params.encoder_type,
    'hidden_dim' : params.hidden_dim,
    'use_cuda': False,

}

# model
encoder_types = ['LSTMEncoder']

assert params.encoder_type in encoder_types, "encoder_type must be in " + str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
batch_size = 50

def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = 0  ## total loss
    correct = 0.0  ##c result
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]


    target_c = train['label'][permutation]

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],word_vec, params.word_emb_dim)
        s1_batch = Variable(s1_batch)


        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        tgt_batch_c = Variable(torch.LongTensor(target_c[stidx:stidx + params.batch_size]))
        k = s1_batch.size(1)

        # model forward
        output = nli_net((s1_batch, s1_len))


        pred = output.data.max(1)[1]  ## c result
        correct += pred.long().eq(tgt_batch_c.data.long()).sum()  ## c result

        # loss
        loss = loss_fn(output, tgt_batch_c)

        all_costs += loss.item()  ## total loss


        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer step
        optimizer.step()

    train_acc = 100 * float(correct)/len(s1) ##c eval
    print('results : epoch {0} ; loss: {1}; mean accuracy train : {2}'.format(epoch, round(all_costs, 2), round(train_acc, 4)))

    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.  ##c result

    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    target_c = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch = Variable(s1_batch)
        tgt_batch_c = Variable(torch.LongTensor(target_c[i:i + params.batch_size]))

        # model forward
        output = nli_net((s1_batch, s1_len))

        pred = output.data.max(1)[1]  ##c result
        correct += pred.long().eq(tgt_batch_c.data.long()).sum()  ##c result

    eval_acc = 100 * float(correct) / len(s1) ##c eval

    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, round(eval_acc, 4)))  # eval_acc
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :{2}'.format(epoch, eval_type, round(eval_acc, 4)))  # eval_acc

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

while epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)