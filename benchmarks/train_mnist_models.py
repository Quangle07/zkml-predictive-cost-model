import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Models

class ModelA_MLP_ReLU(nn.Module):
    """
    Model A: Wide & Linear Heavy
    Fast to prove (ReLU lookup is trivial in EZKL), higher assignment count.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class ModelB_CNN(nn.Module):
    """
    Model B: Spatial & Dense (CNN)
    High parameter spatial model. Uses Conv2d and MaxPool2d.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ModelC_Sigmoid(nn.Module):
    """
    Model C: Non-Linear Heavy
    Uses Sigmoid activation. Isolates the lookup table span (L_span) penalty.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.Sigmoid(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class ModelD_TinyCNN(nn.Module):
    """
    Model D: A lightweight CNN to fill the gap between Model A and Model B.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 28x28 -> 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 14 * 14, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ModelE_DeepMLP(nn.Module):
    """
    Model E: Tests depth vs width by using multiple smaller hidden layers.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


class ModelF_GELU(nn.Module):
    """
    Model F: Tests the lookup table penalty (L_span) of GELU activations.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.GELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class ModelG_MediumCNN(nn.Module):
    """
    Model G: Fills the gap between Tiny CNN (8-channel) and Heavy CNN (32-channel).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ModelH_UltraLightMLP(nn.Module):
    """
    Model H: Pushes the extreme left boundary. How fast can EZKL go?
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

# Training & Evaluation Pipeline

def train_and_eval(model, train_loader, test_loader, epochs=5, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
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

    accuracy = 100.0 * correct / total
    return accuracy


def export_to_onnx(model, model_name, output_dir="onnx_models"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model = model.to("cpu")

    # Standard MNIST input tensor: batch_size=1, channels=1, height=28, width=28
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # Opset 14 is fully supported by EZKL
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Fixed shape for EZKL grid compilation
    )
    print(f"Exported ONNX graph to: {onnx_path}")

# Main Execution

if __name__ == "__main__":
    # Dataset Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data_mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data_mnist', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    models = {
        "model_a_mlp": ModelA_MLP_ReLU(),
        "model_b_cnn": ModelB_CNN(),
        "model_c_sigmoid": ModelC_Sigmoid(),
        "model_d_tinycnn": ModelD_TinyCNN(),
        "model_e_deepmlp": ModelE_DeepMLP(),
        "model_f_gelu": ModelF_GELU(),
        "model_g_mediumcnn": ModelG_MediumCNN(),
        "model_h_ultralightmlp": ModelH_UltraLightMLP()
    }

    results = {}

    print("--- Starting Training and Export ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        acc = train_and_eval(model, train_loader, test_loader, epochs=5)
        results[name] = acc
        print(f"{name} MNIST Accuracy: {acc:.2f}%")

        # Export to ONNX for EZKL
        export_to_onnx(model, name)

    print("\n--- Summary ---")
    for name, acc in results.items():
        print(f"{name}: {acc:.2f}% test accuracy")
