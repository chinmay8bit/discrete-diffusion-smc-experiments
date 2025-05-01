import torch
from torch import Tensor

class NoiseScheduler():
    def __init__(self, schedule: str):
        assert schedule in ['linear', 'cosine']
        self.schedule = schedule
        
    def alpha(self, t: Tensor) -> Tensor:
        """
        Calculates alpha(t)
        """
        if t.dim() == 0:
            assert t >= 0 and t <= 1
        else:
            assert torch.all(t >= 0) and torch.all(t <= 1)
        
        if self.schedule == 'linear':
            return 1 - t
        elif self.schedule == 'cosine':
            return 1 - torch.cos((torch.pi/2) * (1 - t))
        else:
            raise NotImplementedError
        
    def alpha_dash(self, t: Tensor) -> Tensor:
        """
        Calculates alpha'(t)
        """
        if t.dim() == 0:
            assert t >= 0 and t <= 1
        else:
            assert torch.all(t >= 0) and torch.all(t <= 1)

        if self.schedule == 'linear':
            return -1
        elif self.schedule == 'cosine':
            return -(torch.pi/2) * torch.sin((torch.pi/2) * (1 - t))
        else:
            raise NotImplementedError


class DiscreteTimeScheduler():
    def __init__(self, schedule: str, num_timesteps: int):
        assert schedule in ['linear', 'cosine']
        self.schedule = schedule
        self.num_timesteps = num_timesteps

    def discrete_time(self, i):
        assert i >= 0 and i <= self.num_timesteps
        if i == 0:
            return torch.tensor(0)
        if self.schedule == 'linear':
            return torch.tensor(i / self.num_timesteps) 
        elif self.schedule == 'cosine':
            return torch.cos(
                (torch.pi / 2) * (1 - torch.tensor(i / self.num_timesteps))
            )
        else:
            raise ValueError(
                f"Invalid discretization schedule: {self.schedule}"
            )
