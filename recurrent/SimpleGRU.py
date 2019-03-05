import torch

# This class defines a slightly simplified version of the torch.nn.GRU
# network which processes a single time series and includes an output
# layer.

class SimpleGRU(torch.nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        # In the constructor, we first call the constructor
        # for a general neural network module
        super(SimpleGRU, self).__init__()
        # Store the dimensionalities
        self.dim_in  = int(dim_in)
        self.dim_hid = int(dim_hid)
        self.dim_out = int(dim_out)
        # Then we set up the GRU layer
        self.gru = torch.nn.GRU(self.dim_in, self.dim_hid)
        # Further, we set up a linear neural network layer to map
        # the current GRU state to the output
        self.output_layer = torch.nn.Linear(self.dim_hid, self.dim_out)
    
    def forward(self, X, h = None):
        # this computes the forward pass for our GRU.
        # X is a single (!) input sequence in form of a tensor
        # of shape T x dim_in.
        T = X.size()[0]
        if(X.size()[1] != self.dim_in):
            raise ValueError('Expected %d input dimensions but got %d!' % (self.dim_in, X.size()[1]))

        # To make this compatible with the pyTorch GRU implementation,
        # we need to add an empty second dimension to the
        # input, which is reserved to process multiple squences at once
        X = X.unsqueeze(1)

        # Now, execute the GRU layer, which returns the hidden states
        # for all time steps
        H, _ = self.gru(X, h)

        # Then, execute the output layer, which returns the output
        # for all time steps
        Y = self.output_layer(H)

        # And return the version where we removed the middle dimension
        # again.
        return Y.squeeze(dim=1)
