from dataclasses import dataclass
import torch
from torch.autograd.functional import jacobian
from optimization.utils import *

@dataclass(kw_only=True)


class QR(OptimizationParameters):
    Q: torch.Tensor
    Qf: torch.Tensor
    R: torch.Tensor
    Xref: torch.Tensor
    Uref: torch.Tensor

def stage_cost(x, u, k, params: QR):
    # return stage cost at time step k 
    dx = x - params.Xref[k]
    du = u - params.Uref[k]
    return 1/2 * dx.T @ params.Q @ dx + 1/2 * du.T @ params.R @ du

def term_cost(x, params: QR):
    # return terminal cost
    dx = x - params.Xref[-1]
    return 1/2 * dx.T @ params.Qf @ dx

def stage_cost_expansion(x, u, k, params: QR):
    # ∇ₓ²J, ∇ₓJ, ∇ᵤ²J, ∇ᵤJ
    dx = x - params.Xref[k]
    du = u - params.Uref[k]

    return params.Q, params.Q @ dx, params.R, params.R @ du

def term_cost_expansion(x, params: QR):
    # ∇ₓ²Jn, ∇ₓJn
    dx = x - params.Xref[-1]

    return params.Qf, params.Qf @ dx

def backward_pass(X, U, fd, params: QR):
    nx, nu, N = params.nx, params.nu, params.N 
    
    # vectors of vectors/matrices for recursion 
    P = torch.zeros((N, nx, nx)) # cost to go quadratic term
    p = torch.zeros((N, nx)) # cost to go linear term
    d = torch.zeros((N-1, nu)) # feedforward control
    K = torch.zeros((N-1, nu, nx)) # feedback gain

    # implement backwards pass and return d, K, ΔJ 
    N = params.N
    ΔJ = 0.0

    hn, gn = term_cost_expansion(X[-1], params)
    
    # Gradients and hessians for cost-to-go value function
    P[-1] = hn
    p[-1] = gn
    for k in range(N-2, -1, -1):
        hx, gx, hu, gu = stage_cost_expansion(X[k], U[k], k, params)
        
        # Find df/dx and df/du
        # Linearize about the reference trajectory
        A = jacobian(lambda _x: fd(_x, U[k], params), X[k])
        B = jacobian(lambda _u: fd(X[k], _u, params), U[k])

        # Add to action-value function derivates the derivatives of value function
        gx = gx + A.T @ p[k+1]
        gu = gu + B.T @ p[k+1]
    
        # Add to action-value function hessians the hessians of value function
        Gxx = hx + A.T @ P[k+1] @ A 
        Guu = hu + B.T @ P[k+1] @ B 
        Gxu = A.T @ P[k+1] @ B # Stage cost with 0 d/dxdu
        Gux = B.T @ P[k+1] @ A # Stage cost with 0 d/dudx
        
        d[k] = torch.linalg.solve(Guu, gu)
        K[k] = torch.linalg.solve(Guu, Gux)
        
        P[k] = Gxx + K[k].T @ Guu @ K[k] - Gxu @ K[k] - K[k].T @ Gux
        p[k] = gx - K[k].T @ gu + K[k].T @ Guu @ d[k] - Gxu @ d[k]
            
        ΔJ += gu.T @ d[k]
    
    return d, K, ΔJ

def trajectory_cost(X, U, params: QR):
    N = params.N

    cost = 0
    for k in range(N-1):
        cost += stage_cost(params, X[k], U[k], k)
    
    cost += term_cost(params, X[-1])
    
    return cost 

def forward_pass(X, U, d, K, fd, params: QR, max_linesearch_iters = 20):
    nx, nu, N = params.nx, params.nu, params.N 
    
    Xn = torch.zeros((N, nx)) # new state history 
    Un = torch.zeros((N-1, nu)) # new control history 
    
    # initial condition 
    Xn[0] = 1*X[0]
    
    # initial step length 
    α = 1.0
    
    initial_cost = trajectory_cost(params, X, U)
    
    # forward pass 
    for i in range(max_linesearch_iters):
        for k in range(N-1):
            Δx = Xn[k] - X[k]
            Δu = -α*d[k] - K[k]*Δx
            Un[k] = U[k] + Δu
            Xn[k+1] = fd(params, Xn[k], Un[k], k)
        
        cost = trajectory_cost(params, Xn, Un)
        if cost < initial_cost:
            return Xn, Un, cost, α

        α = α/2

    raise RuntimeError("forward pass failed")

def iLQR(x0, U, fd, params: QR, atol=1e-3, max_iters = 250):
    assert U.size(0) == params.N-1
    assert U.size(1) == params.nu
    assert x0.size() == (params.nx,)

    nx, nu, N = params.nx, params.nu, params.N

    # initial rollout
    X = torch.zeros((N, nx))
    X[0] = x0
    for k in range(N-1):
        X[k+1] = fd(X[k], U[k], params)

    for ilqr_iter in range(max_iters):
        d, K, ΔJ = backward_pass(X, U, fd, params)
        
        X, U, J, α = forward_pass(X, U, d, K, fd, params)
        
        # termination criteria 
        if ΔJ < atol:
            print(f"iLQR converged in {ilqr_iter+1} iteration(s)")
            return X, U, K 

    raise RuntimeError("iLQR failed")
