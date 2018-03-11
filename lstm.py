import mxnet as mx
from mxnet import nd
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import time
import math

import sys
sys.path.append('.')
import os
import numpy as np
import utils
import chord_preprocess

ctx = utils.try_gpu()
print('Will use', ctx)

class RNNModel(gluon.Block):
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, embed_dim, weight_initializer=mx.init.Uniform(0.1))

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='relu', dropout=dropout, input_size=embed_dim)
            
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)
            
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)

            elif mode == 'gru':
                self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)
            
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, rnn_tanh, lstm and grb"%mode)
            
            self.decoder = nn.Dense(vocab_size, in_units=hidden_dim)
            self.hidden_dim = hidden_dim
        
    def forward(self, inputs, state):
        emb = self.drop(self.encoder(inputs))
        output, state = self.rnn(emb, state)
        output = self.drop(output)
        decodeed = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decodeed, state
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class Dictionary:

    def __init__(self):
        self.note_to_idx = {}
        self.idx_to_note = []
    
    def add_note(self, note):
        if note not in self.note_to_idx:
            self.idx_to_note.append(note)
            self.note_to_idx[note] = len(self.idx_to_note) - 1
        return self.note_to_idx[note]
    
    def __len__(self):
        return len(self.idx_to_note)

class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()

        data = self.tokenize(path)
        train_size = int(len(data) * 0.8)
        self.train = data[:train_size]
        self.valid = data[train_size + 1 :]


    def tokenize(self, path):
        assert os.path.exists(path)

        files = os.listdir(path)

        midi_files = [file for file in files if file.split('.')[1] == 'mid']

        total_notes = []
        for midi in midi_files:
            notes = chord_preprocess.parse_midi(path + '/' + midi)
            for note in notes:
                total_notes.append(note)
                self.dictionary.add_note(note)
        
        indices = np.zeros( (len(total_notes),) , dtype='int32')
        for idx, note in enumerate(total_notes):
            indices[idx] = self.dictionary.note_to_idx[note]
        
        return mx.nd.array(indices, dtype='int32') 


def batchify(data, batch_size):
    num_batchs = data.shape[0] // batch_size
    data = data[:num_batchs * batch_size]
    data = data.reshape((batch_size, num_batchs)).T
    return data

corpus = None

model_name = 'lstm'

embed_dim = 100
hidden_dim = 100
num_layers = 2
lr = 1.0
clipping_norm = 0.2
epochs = 1
batch_size = 32
num_steps = 5
dropout_rate = 0.2
eval_period = 500

vocab_size = len(corpus.dictionary)

train_data = batchify(corpus.train, batch_size).as_in_context(ctx)
val_data = batchify(corpus.valid, batch_size).as_in_context(ctx)
test_data = batchify(corpus.test, batch_size).as_in_context(ctx)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate)

model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 -i)
    data = source[i: i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state

def model_eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(fun=mx.nd.zeros, batch_size=batch_size, ctx=ctx)

    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train():
    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(fun = mx.nd.zeros, batch_size=batch_size, ctx=ctx)
        
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, num_steps)):
            data, target = get_batch(train_data, i)

            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()
            
            grads = [i.grad(ctx) for i in model.collect_params().values()]

            gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0
            
        val_L = model_eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation '
              'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
                                   math.exp(val_L)))


train()
test_L = model_eval(test_data)
print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))

