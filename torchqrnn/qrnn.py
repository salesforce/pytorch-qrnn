import torch
from torch import nn
from torch.autograd import Variable

if __name__ == '__main__':
    from forget_mult import ForgetMult
else:
    from .forget_mult import ForgetMult


class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
        bidirectional: If True, becomes a bidirectional layer (num_directions will be 2). Default: False.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (num_directions, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, num_directions * hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True, use_cuda=True, bidirectional=False):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate
        self.use_cuda = use_cuda
        self.num_directions = 2 if bidirectional else 1

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size * self.num_directions if self.output_gate else 2 * self.hidden_size * self.num_directions)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size * self.num_directions)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size * self.num_directions)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.nn.functional.tanh(Z)
        F = torch.nn.functional.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # For the bidirectional case, split into fwd and bwd and compute bwd.
        # There should be a nicer way to do this.
        if self.num_directions == 2:
            Z, Z_bwd = Z.chunk(2, dim=2)
            F, F_bwd = F.chunk(2, dim=2)
            O, O_bwd = O.chunk(2, dim=2)
            Z_bwd = Z_bwd.contiguous()
            F_bwd = F_bwd.contiguous()
            C_bwd = ForgetMult(backwards=True)(F_bwd, Z_bwd, hidden[1] if hidden is not None else None, use_cuda=self.use_cuda)
            if self.output_gate:
                H_bwd = torch.nn.functional.sigmoid(O_bwd) * C_bwd
            else:
                H_bwd = C_bwd

        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel
        Z = Z.contiguous()
        F = F.contiguous()
        C = ForgetMult()(F, Z, hidden[0] if hidden is not None else None, use_cuda=self.use_cuda)

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.nn.functional.sigmoid(O) * C
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        if self.num_directions == 2:
            h_n_fwd = C[-1:, :, :]
            h_n_bwd = C_bwd[:1, :, :]
            h_n = torch.cat([h_n_fwd, h_n_bwd], dim=0)
            return torch.cat([H, H_bwd], dim=2), h_n
        else:
            return H, C[-1:, :, :]



class QRNN(torch.nn.Module):
    r"""Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        num_layers: The number of QRNN layers to produce.
        layers: List of preconstructed QRNN layers to use for the QRNN module (optional).
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
        bidirectional: If True, becomes a bidirectional QRNN (num_directions=2). Default: False.

    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for the QRNN.

    Outputs: output, h_n
        - output (seq_len, batch, num_directions * hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, layers=None, **kwargs):
        assert batch_first == False, 'Batch first mode is not yet supported'
        assert bias == True, 'Removing underlying bias is not yet supported'

        super(QRNN, self).__init__()

        self.num_directions = 2 if bidirectional else 1
        self.layers = torch.nn.ModuleList(layers if layers else [QRNNLayer(input_size if l == 0 else hidden_size * self.num_directions, hidden_size, bidirectional=bidirectional, **kwargs) for l in range(num_layers)])
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers) if layers else num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout


    def reset(self):
        r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
        [layer.reset() for layer in self.layers]

    def forward(self, input, hidden=None):
        next_hidden = []
        _, batch_size, _ = input.size()

        for i, layer in enumerate(self.layers):
            # for bidirectional (num_directions = 2): 0, 2, 4, 6...
            # else (num_directions = 1): 0, 1, 2, 3...
            hidden_offset = self.num_directions * i
            input, hn = layer(input, None if hidden is None else hidden[hidden_offset:hidden_offset + self.num_directions])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = torch.nn.functional.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers * self.num_directions, batch_size, self.hidden_size)

        return input, next_hidden


if __name__ == '__main__':

    print('Test bidirectional')
    seq_len, batch_size, hidden_size, input_size = 7, 20, 256, 32
    size = (seq_len, batch_size, input_size)
    X = torch.autograd.Variable(torch.rand(size), requires_grad=True).cuda()
    num_layers = 3
    h = torch.autograd.Variable(torch.rand(num_layers * 2, batch_size, hidden_size).cuda(), requires_grad=True)
    qrnn = QRNN(input_size, hidden_size, num_layers=num_layers, dropout=0.4, bidirectional=True)
    qrnn.cuda()
    output, hidden = qrnn(X, h)
    assert list(output.size()) == [7, 20, 256 * 2]
    assert list(hidden.size()) == [num_layers * 2, 20, 256]

    ###
    seq_len, batch_size, hidden_size, input_size = 35, 8, 32, 32
    size = (seq_len, batch_size, input_size)
    X = Variable(torch.rand(size), requires_grad=True).cuda()

    qrnn = QRNNLayer(input_size, hidden_size, bidirectional=True)
    qrnn.cuda()
    Y, _ = qrnn(X)

    qrnn.use_cuda = False
    Z, _ = qrnn(X)

    diff = (Y - Z).sum().item()
    print('Total difference between QRNN(bidirectional=True, use_cuda=True) and QRNN(bidirectional=True, use_cuda=False) results:', diff)
    assert diff < 1e-5, 'CUDA and non-CUDA QRNN layers return different results'

    from torch.autograd import gradcheck
    seq_len, batch_size, hidden_size, input_size = 4, 8, 16, 16
    size = (seq_len, batch_size, input_size)
    X = Variable(torch.rand(size), requires_grad=True).cuda()
    inputs = [X,]
    # Had to loosen gradient checking, as for forget mult
    test = gradcheck(QRNNLayer(input_size, hidden_size, output_gate=True, bidirectional=True).cuda(), inputs, eps=1e-4, atol=1e-2)
    print('Gradient check', test)

    print('Test unidirectional')

    seq_len, batch_size, hidden_size, input_size = 7, 20, 256, 32
    size = (seq_len, batch_size, input_size)
    X = torch.autograd.Variable(torch.rand(size), requires_grad=True).cuda()
    qrnn = QRNN(input_size, hidden_size, num_layers=2, dropout=0.4)
    qrnn.cuda()
    output, hidden = qrnn(X)
    assert list(output.size()) == [7, 20, 256]
    assert list(hidden.size()) == [2, 20, 256]

    ###

    seq_len, batch_size, hidden_size = 2, 2, 16
    seq_len, batch_size, hidden_size = 35, 8, 32
    size = (seq_len, batch_size, hidden_size)
    X = Variable(torch.rand(size), requires_grad=True).cuda()

    qrnn = QRNNLayer(hidden_size, hidden_size)
    qrnn.cuda()
    Y, _ = qrnn(X)

    qrnn.use_cuda = False
    Z, _ = qrnn(X)

    diff = (Y - Z).sum().item()
    print('Total difference between QRNN(use_cuda=True) and QRNN(use_cuda=False) results:', diff)
    assert diff < 1e-5, 'CUDA and non-CUDA QRNN layers return different results'

    from torch.autograd import gradcheck
    inputs = [X,]
    # Had to loosen gradient checking, as for forget mult
    test = gradcheck(QRNNLayer(hidden_size, hidden_size).cuda(), inputs, eps=1e-4, atol=1e-2)
    print('Gradient check', test)

