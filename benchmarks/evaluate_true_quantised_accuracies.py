import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the 8 Models (same as before)
class ModelA_MLP_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x): return self.net(x)

class ModelB_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 14 * 14, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class ModelC_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.Sigmoid(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

class ModelD_TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * 14 * 14, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class ModelE_DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x): return self.net(x)

class ModelF_GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.GELU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

class ModelG_MediumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 14 * 14, 10))
    def forward(self, x): return self.classifier(self.features(x))

class ModelH_UltraLightMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x): return self.net(x)

# Quantisation
def simulate_ezkl_quantisation(model, bits, scale):
    """Simulates EZKL's fixed-point quantisation on model weights."""
    q_model = copy.deepcopy(model)
    qmin = -(2**(bits-1))
    qmax = (2**(bits-1)) - 1
    scale_factor = 2**scale

    with torch.no_grad():
        for param in q_model.parameters():
            # Scale, round to nearest integer, and clip to bit constraints
            q_param = torch.round(param * scale_factor)
            q_param = torch.clamp(q_param, qmin, qmax)
            # Rescale back to floats for PyTorch to evaluate
            param.copy_(q_param / scale_factor)

    return q_model

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    models = {
        "Model A": ModelA_MLP_ReLU(), "Model B": ModelB_CNN(), "Model C": ModelC_Sigmoid(),
        "Model D": ModelD_TinyCNN(), "Model E": ModelE_DeepMLP(), "Model F": ModelF_GELU(),
        "Model G": ModelG_MediumCNN(), "Model H": ModelH_UltraLightMLP()
    }

    bit_scales = [(16, 7), (12, 5), (8, 3), (6, 2), (4, 1)]

    print(f"{'Configuration':<25} | {'Bits':<5} | {'True Accuracy (%)'}")
    print("-" * 55)

    for name, model in models.items():
        # Train baseline
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                criterion(model(data), target).backward()
                optimizer.step()

        # Evaluate all quantisation tiers
        for bits, scale in bit_scales:
            if bits == 16:
                acc = evaluate(model, test_loader)
            else:
                q_model = simulate_ezkl_quantisation(model, bits, scale)
                acc = evaluate(q_model, test_loader)

            print(f"{name:<25} | {bits:<5} | {acc:.2f}%")
