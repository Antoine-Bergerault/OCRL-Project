from dataclasses import dataclass
import torch
from optimization.utils import *

@dataclass(kw_only=True)
class BicycleModelParameters(ModelParameters):
    min_steering_angle: float
    max_steering_angle: float
    wheelbase: float

normalise_angle = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))

def wrap(x, y, yaw, velocity):
    if x.numel() > 1:
    	return torch.cat([x,y,yaw,velocity], dim=1)	
    return torch.tensor([x, y, yaw, velocity])

def unwrap(x):
    if len(x.shape) > 1:
        return x[:,[0]], x[:,[1]], x[:,[2]], x[:,[3]]   	
    return x[0], x[1], x[2], x[3]

def dynamics(x, u, params: BicycleModelParameters):
    x, y, yaw, velocity = unwrap(x)
    if len(x.shape) > 1:
    	acceleration, steering_angle = u[:,[0]], u[:,[1]]
    else:
    	acceleration, steering_angle = u[0], u[1]
    x_dot = velocity*torch.cos(yaw)
    y_dot = velocity*torch.sin(yaw)

    yaw_dot = velocity*torch.tan(steering_angle) / params.wheelbase

    v_dot = acceleration
    return wrap(x_dot, y_dot, yaw_dot, v_dot)

def safe(x):
    x[2] = normalise_angle(x[2])
    
    return x

def discrete_dynamics(x, u, params: OptimizationParameters):
    """	
    TODO: Complete doc
    """
    return safe(midpoint(dynamics, x, u, params.model))
