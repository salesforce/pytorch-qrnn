import time

import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hidden_size=1024, parallel=True, layers=3, vocab=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab, hidden_size)

        from torchqrnn import QRNN
        self.rnn = QRNN(hidden_size, hidden_size, num_layers=layers)
        #self.rnn = nn.LSTM(hidden_size, hidden_size)
        # Note: we tell DataParallel to split on the second dimension as RNNs are batch second by default in PyTorch
        if parallel: self.rnn = nn.DataParallel(self.rnn, dim=1)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        return out[:-1]

H = 256
SEQ = 100
BATCH = 64

H = 1024
SEQ = 500
BATCH = 128

LOOPS = 500

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

x = torch.autograd.Variable(torch.LongTensor(np.random.randint(0, 100, [BATCH, SEQ])))
x = x.cuda()

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print('Single')
model = Model(H, parallel=False)
model = model.cuda()
# Call once to compile CUDA kernel / set up new GPUs
model(x)
start = time.time()
for _ in range(LOOPS): y = model(x)
print('Time:', time.time() - start)
del model

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print('Multi')
model = Model(H, parallel=True)
model = model.cuda()
# Call once to compile CUDA kernel / set up new GPUs
model(x)
start = time.time()
for _ in range(LOOPS): y2 = model(x)
print('Time:', time.time() - start)

print('Difference:')
print((y - y2).sum())
