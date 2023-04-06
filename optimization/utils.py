from dataclasses import dataclass
import torch

@dataclass(kw_only=True)
class ModelParameters:
    dt: float

@dataclass(kw_only=True)
class OptimizationParameters:
    nx: int
    nu: int
    N: int
    model: ModelParameters

def midpoint(f, x, u, params: ModelParameters):
    x_m = x + params.dt/2 * f(x, u, params) # we assume 0th-order hold
    return x + params.dt * f(x_m, u, params)

def simulate(x0, U, fd, params: OptimizationParameters):
    assert x0.size(0) == params.nx

    X = torch.zeros(params.N+1, params.nx)
    X[0] = x0

    # Simulate dynamics
    for i in range(params.N-1):
        x = X[i]
        u = U[i]
        X[i+1] = fd(x, u, params)

    return X