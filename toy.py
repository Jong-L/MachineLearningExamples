import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(torch.max(x, dim=1))