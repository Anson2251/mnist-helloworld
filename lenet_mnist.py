import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

class Conv(nn.Module):
    def __init__(self,
        ch_in: int,
        ch_out: int,
        kernal_size: tuple[int, int] = (3, 3),
        act: nn.Module | None = None,
        bn: bool = True
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernal_size)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = act if act else nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

class Linear(nn.Module):
    def __init__(self,
        feat_in: int,
        feat_out: int,
        bias: bool = True,
        act: nn.Module | None = None,
        dropout: float = 0.5
    ):
        super(Linear, self).__init__()
        self.linear = nn.Linear(feat_in, feat_out, bias)
        self.act = act if act else nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.drop(self.act(self.linear(x)))

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.channels: int = 16
        self.features: nn.Sequential = nn.Sequential(
            Conv(1, self.channels, (5, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(self.channels, self.channels, (5, 1)),
            Conv(self.channels, self.channels, (1, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier: nn.Sequential = nn.Sequential(
            Linear(self.channels*4*4, self.channels*4*2, dropout=0.2),  # assume 28x28
            Linear(self.channels*4*2, self.channels*4, dropout=0.3),
            nn.Linear(self.channels*4, 10)
        )


    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(-1, self.channels*4*4)
        x = self.classifier(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(-1, 16*4*4)
        x = self.classifier(x)
        return x

def display_progress(
    epoch: int,
    epochs: int,
    batch_idx: int | None,
    total_batches: int,
    loss: float,
    accuracy: float | None = None,
    phase: str = "train"
):
    """Display training/validating progress"""
    progress_str = f"Epoch [{epoch+1}/{epochs}] {phase.capitalize()} "
    if batch_idx is not None:
        progress_str += f"Batch [{batch_idx+1}/{total_batches}] "
    progress_str += f"Loss: {loss:.4f}"
    if accuracy is not None:
        progress_str += f" | Acc: {accuracy:.2f}%"
    return progress_str

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_accuracy': accuracy
    }
    torch.save(checkpoint, filepath)

def manage_checkpoints(
    checkpoint_dir: str,
    current_checkpoint_path: str,
    best_checkpoint_path: str,
    current_acc: float,
    best_acc: float
):
    """Keep only the most recent and best checkpoints"""
    # Remove old checkpoints except current and best
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and file not in [os.path.basename(current_checkpoint_path), os.path.basename(best_checkpoint_path)]:
            try:
                os.remove(os.path.join(checkpoint_dir, file))
            except FileNotFoundError:
                pass

    # Update best checkpoint if current is better
    if current_acc > best_acc:
        if os.path.exists(best_checkpoint_path):
            os.remove(best_checkpoint_path)
        # Copy current to best
        torch.save(torch.load(current_checkpoint_path), best_checkpoint_path)
        return current_acc
    return best_acc

def train_model(
    model: nn.Module,
    train_loader: DataLoader[torchvision.datasets.MNIST],
    val_loader: DataLoader[torchvision.datasets.MNIST],
    criterion: CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    checkpoint_dir: str = 'checkpoints'
):
    _ = model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_acc = 0.0
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')

        for i, (images, labels) in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100 * correct / total
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)

        epoch_acc = test_model(model, val_loader, device)
        model.train()

        # Save current checkpoint
        current_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        save_checkpoint(model, optimizer, epoch+1, epoch_loss, epoch_acc, current_checkpoint_path)

        # Manage checkpoints (keep only latest and best)
        best_acc = manage_checkpoints(checkpoint_dir, current_checkpoint_path, best_checkpoint_path, epoch_acc, best_acc)

    return epochs, best_acc

def test_model(
    model: nn.Module,
    test_loader: DataLoader[torchvision.datasets.MNIST],
    device: torch.device
):
    _ = model.eval()
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc='Validating')

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100 * correct / total
            pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})

    final_acc = 100 * correct / total
    return final_acc

def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None
):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        acc = checkpoint.get('train_accuracy', 0.0)
        print(f"  Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
        return epoch, loss, acc
    else:
        return 0, 0.0, 0.0

training_epoch = 20
learning_rate = 1e-3

def main():
    using_cpu = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        using_cpu = True
        device = torch.device('cpu')
    print(f'Using device: {str(device).upper()}')

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1326,), (0.3106,))
    ])

    num_of_cpus = os.cpu_count()
    worker_num_unit = (num_of_cpus if num_of_cpus else 1) // 3
    train_worker_num = max(1, worker_num_unit*2) if not using_cpu else 1
    val_worker_num = max(1, worker_num_unit) if not using_cpu else 1

    print(f"Using {train_worker_num} CPUs for training data loader, {val_worker_num} CPUs for validating data loader")

    print("Loading MNIST dataset")
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=val_transform)

    train_loader: DataLoader[torchvision.datasets.MNIST] = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=train_worker_num)
    test_loader: DataLoader[torchvision.datasets.MNIST] = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=val_worker_num)

    model = MyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print('Starting training\n')
    start_time = time.time()
    epochs, best_acc = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=training_epoch)
    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nSummary: Epoch: {epochs}, Best Training Accuracy: {best_acc:.2f}%")
    print(f'Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)')

    best_checkpoint_path = os.path.join('checkpoints', 'best_model.pt')

    # Load the best checkpoint for validating
    if os.path.exists(best_checkpoint_path):
        print("Best model info:")
        _ = load_checkpoint(best_checkpoint_path, model)
    else:
        print("No best checkpoint found")
        return

    # Also save just the state dict for compatibility
    torch.save(model.state_dict(), 'lenet_mnist.pth')
    print('Model state dict saved as lenet_mnist.pth')

if __name__ == '__main__':
    main()
