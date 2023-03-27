from collections import namedtuple
import torch

Params = namedtuple('Params', ['max_steering_angle', 'min_steering_angle', 'dt'])

normalise_angle = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))

def discrete_dynamics(x, y, yaw, velocity, acceleration, steering_angle, params, dt):
    """
    TODO: Complete doc
    """
    # Compute the local velocity in the x-axis
    new_velocity = velocity + dt * acceleration

    # Limit steering angle to physical vehicle limits
    steering_angle = torch.clamp(steering_angle, -self.max_steer, self.max_steer)

    # Compute the angular velocity
    angular_velocity = new_velocity*torch.tan(steering_angle) / self.wheelbase

    # Compute the final state using the discrete time model
    new_x   = x + velocity*torch.cos(yaw)*dt
    new_y   = y + velocity*torch.sin(yaw)*dt
    new_yaw = normalise_angle(yaw + angular_velocity * dt)
    
    return new_x, new_y, new_yaw, new_velocity, steering_angle, angular_velocity