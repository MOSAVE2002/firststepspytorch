import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y*y*2
print(z)
# z = z.mean()
# print(z)

# Backpropagation 
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32) # Jacobian vector product
z.backward(v) # dz/dx
# z.backward() # dz/dx
print(x.grad)

# ----------------------------------------------
x = torch.randn(3, requires_grad=True)
print(x)
# x.requires_grad_(False) # In-place operation to change requires_grad to False
# x.detach() # Detach x from the computation graph, so that it will not be tracked for gradients
#with torch.no_grad(): # Temporarily set all the requires_grad flags to false
x.requires_grad_(False) # In-place operation to change requires_grad to False
y = x.detach() # Detach y from the computation graph, so that it will not be tracked for gradients
with torch.no_grad(): # Temporarily set all the requires_grad flags to false
    y = x + 2
    print(y)

# dummy training example
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_() # Manually zero the gradients after each epoch

optimizer = torch.optim.SGD(weights, lr=0.01) # SGD = Stochastic Gradient Descent ; lr = learning rate
optimizer.step() # Update the weights based on the gradients
optimizer.zero_grad() # Manually zero the gradients after each epoch