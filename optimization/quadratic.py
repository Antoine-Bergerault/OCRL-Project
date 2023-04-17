from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from optimization.utils import *

@dataclass(kw_only=True)
class QR(OptimizationParameters):
    Q: npt.NDArray
    Qf: npt.NDArray
    R: npt.NDArray
    Xref: npt.NDArray
    Uref: npt.NDArray

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

def backward_pass(X, U, fd_grad, params: QR):
    nx, nu, N = params.nx, params.nu, params.N 
    
    # vectors of vectors/matrices for recursion 
    P = np.zeros((N, nx, nx)) # cost to go quadratic term
    p = np.zeros((N, nx)) # cost to go linear term
    d = np.zeros((N-1, nu)) # feedforward control
    K = np.zeros((N-1, nu, nx)) # feedback gain

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
        grad = fd_grad(X[k], U[k], params)
        A = grad[0]
        B = grad[1]

        # Add to action-value function derivates the derivatives of value function
        gx = gx + A.T @ p[k+1]
        gu = gu + B.T @ p[k+1]
    
        # Add to action-value function hessians the hessians of value function
        Gxx = hx + A.T @ P[k+1] @ A 
        Guu = hu + B.T @ P[k+1] @ B 
        Gxu = A.T @ P[k+1] @ B # Stage cost with 0 d/dxdu
        Gux = B.T @ P[k+1] @ A # Stage cost with 0 d/dudx
        
        d[k] = np.linalg.solve(Guu, gu)
        K[k] = np.linalg.solve(Guu, Gux)
        
        P[k] = Gxx + K[k].T @ Guu @ K[k] - Gxu @ K[k] - K[k].T @ Gux
        p[k] = gx - K[k].T @ gu + K[k].T @ Guu @ d[k] - Gxu @ d[k]
            
        ΔJ += gu.T @ d[k]
    
    return d, K, ΔJ

def trajectory_cost(X, U, params: QR):
    N = params.N

    cost = 0
    for k in range(N-1):
        cost += stage_cost(X[k], U[k], k, params)
    
    cost += term_cost(X[-1], params)
    
    return cost 

def forward_pass(X, U, d, K, fd, params: QR, max_linesearch_iters = 20):
    nx, nu, N = params.nx, params.nu, params.N 
    
    Xn = np.zeros((N, nx)) # new state history 
    Un = np.zeros((N-1, nu)) # new control history 
    
    # initial condition 
    Xn[0] = 1*X[0]
    
    # initial step length 
    α = 1.0
    
    initial_cost = trajectory_cost(X, U, params)
    
    # forward pass 
    for i in range(max_linesearch_iters):
        for k in range(N-1):
            Δx = Xn[k] - X[k]
            Δu = -α*d[k] - K[k] @ Δx
            Un[k] = U[k] + Δu
            Xn[k+1] = fd(Xn[k], Un[k], params)
        
        cost = trajectory_cost(Xn, Un, params)
        if cost <= initial_cost:
            return Xn, Un, cost, α

        α = α/2

    raise RuntimeError("forward pass failed")

def iLQR(x0, U, fd, fd_grad, params: QR, atol=1e-3, max_iters = 100):
    assert U.shape[0] == params.N-1
    assert U.shape[1] == params.nu
    assert x0.shape == (params.nx,)

    nx, nu, N = params.nx, params.nu, params.N

    # initial rollout
    X = simulate(x0, U, fd, params)

    for ilqr_iter in range(max_iters):
        d, K, ΔJ = backward_pass(X, U, fd_grad, params)
        
        X, U, J, α = forward_pass(X, U, d, K, fd, params)
        
        # termination criteria 
        if ΔJ < atol:
            print(f"iLQR converged in {ilqr_iter+1} iteration(s)")
            return X, U, K 

    raise RuntimeError("iLQR failed")