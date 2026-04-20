import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define model
# Klasse erstellen, erbt von nn.Module, damit wir die Methoden und Eigenschaften von nn.Module nutzen können.
class NeuralNetwork(nn.Module):
    def __init__(self):
        # In the __init__ method, we define the layers of our neural network. 
        # We will use a simple feedforward neural network with two hidden layers.

        super(NeuralNetwork, self).__init__() # super() is used to call the __init__ method of the parent class (nn.Module) to initialize the base class.   
        self.flatten = nn.Flatten()
        # Architectur of the neural network:
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # The input layer has 28*28 neurons , 512 neurons in the first hidden layer, 512 neurons in the second hidden layer and 10 neurons in the output layer (one for each class).
            nn.ReLU(),
            nn.Linear(512, 512), # second hidden layer with 512 neurons
            nn.ReLU(),
            nn.Linear(512, 10) # output layer with 10 neurons (one for each class)
        )

    def forward(self, x):
        x = self.flatten(x)
        # The forward method defines the forward pass of the neural network. 
        # It takes an input tensor x and passes it through the layers defined in the __init__ method.
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # set the model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # move the data to the device (GPU or CPU)

        # Compute prediction error
        pred = model(X) # forward pass
        loss = loss_fn(pred, y) # compute the loss

        # Backpropagation
        optimizer.zero_grad() # zero the gradients before backpropagation
        loss.backward() # backpropagation
        optimizer.step() # update the weights

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # set the model to evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad(): # disable gradient calculation for evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # move the data to the device (GPU or CPU)
            pred = model(X) # forward pass
            test_loss += loss_fn(pred, y).item() # compute the loss and accumulate it
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # compute the number of correct predictions

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


if __name__ == "__main__":
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}") # N is batch size, C is number of channels, H is height and W is width
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    # model = NeuralNetwork().to(device) # move the model to the device (GPU or CPU)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # epochs = 5
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # print("Done!")

    # # Saving the model
    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")

    # Loading the model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))
    
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    print(test_data[0][0])
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

     