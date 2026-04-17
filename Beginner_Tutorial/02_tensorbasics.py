import torch
import numpy as np

x = torch.empty(2, 3, dtype=torch.int)

y = torch.zeros(2, 3)

z = torch.rand(2, 3)

p = torch.tensor ([[1, 2], [3, 4]])

x = torch.rand(2, 2)
y = torch.rand(2,2)
z = x + y
z = torch.add(x, y)
z = x - y
z = torch.sub(x, y)
z = x * y
z = torch.mul(x, y)
z = x / y
z = torch.div(x, y)

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# Attention: a and b share the same memory. If we change a, b will change as well. Only if it is a CPU Tensor, it can be converted to a NumPy array and vice versa.

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu") # Change z back to CPU if needed




# print(x.dtype)
# print(x.size())
# print(x)