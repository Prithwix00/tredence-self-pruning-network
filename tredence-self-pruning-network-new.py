import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================== PrunableLinear Layer ======================
class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.gate_scores, 0.0)  # sigmoid(0) ≈ 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# ====================== Network ======================
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ====================== Helpers ======================
def get_sparsity_loss(model: nn.Module) -> torch.Tensor:
    return sum(
        torch.sigmoid(module.gate_scores).sum()
        for module in model.modules()
        if isinstance(module, PrunableLinear)
    )

def calculate_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    total = 0
    pruned = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            num = gates.numel()
            total += num
            pruned += (gates < threshold).sum().item()
    return (pruned / total * 100) if total > 0 else 0.0

def train_model(model, trainloader, optimizer, criterion, lambda_val, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            class_loss = criterion(outputs, labels)
            sparsity_loss = get_sparsity_loss(model)
            total_loss = class_loss + lambda_val * sparsity_loss
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f}")

def evaluate_model(model, testloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"  Test Accuracy: {acc:.2f}%")
    return acc

def plot_gate_distribution(model, filename="gate_distribution.png"):
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).cpu().detach().numpy().flatten()
            all_gates.extend(gates)
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=100, color='skyblue', edgecolor='black')
    plt.title("Distribution of Final Gate Values (Best Model)")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.axvline(x=0.01, color='red', linestyle='--', label='Pruning Threshold (1e-2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f" Plot saved as '{filename}'")

# ====================== Main ======================
if __name__ == "__main__":
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    lambda_values = [0.0001, 0.001, 0.01]   # low, medium, high
    num_epochs = 15
    results = []
    best_model = None
    best_acc = -1
    best_lambda = None

    for lam in lambda_values:
        print(f"\n=== Training with λ = {lam} ===")
        model = SelfPruningNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, trainloader, optimizer, criterion, lam, num_epochs, device)
        acc = evaluate_model(model, testloader, device)
        sparsity = calculate_sparsity(model)

        results.append({"lambda": lam, "accuracy": acc, "sparsity": sparsity})
        print(f"  Sparsity Level: {sparsity:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_lambda = lam

    # Results table
    print("\n=== RESULTS SUMMARY ===")
    print("| Lambda | Test Accuracy (%) | Sparsity Level (%) |")
    print("|--------|-------------------|--------------------|")
    for r in results:
        print(f"| {r['lambda']:<6} | {r['accuracy']:.2f}             | {r['sparsity']:.2f}              |")

    print(f"\nBest model (highest accuracy) → λ = {best_lambda} ({best_acc:.2f}%)")
    plot_gate_distribution(best_model)

    print("\n Training finished! Copy the table above into report.md")