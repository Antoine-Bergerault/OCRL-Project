import torch

def forward_pass(x0, U, fd, params):
    X = torch.zeros(U.size(0)+1, x0.size(0))
    X[0] = x0
    # Simulate dynamics
    for i in range(U.size(0)):
        x = X[i]
        u = U[i]
        X[i+1] = fd(x, u, params)

    return X