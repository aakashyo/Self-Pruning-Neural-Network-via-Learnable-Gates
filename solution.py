from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import csv
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(batch_size: int = 128):
    """
    Downloads and splits the CIFAR-10 dataset.
    Returns: train_loader, val_loader, test_loader
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # 80/20 train/validation split
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def prepare_results_dir():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

def plot_experiment_curves(history: dict, lam: float):
    """Plots epoch-wise metrics for a single experiment."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 1 & 2. Loss vs Epoch
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Loss vs Epoch (λ={lam})')
    plt.legend()
    
    # 3 & 4. Accuracy vs Epoch
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy vs Epoch (λ={lam})')
    plt.legend()
    
    # 5. Sparsity vs Epoch
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['overall_sparsity'], label='Overall Sparsity')
    plt.title(f'Sparsity vs Epoch (λ={lam})')
    plt.legend()
    
    # 6. Classification vs Sparsity Loss
    plt.subplot(2, 3, 4)
    plt.scatter(history['cls_loss'], history['sparsity_loss'], c=epochs, cmap='viridis')
    plt.xlabel('Classification Loss')
    plt.ylabel('Sparsity Loss')
    plt.colorbar(label='Epoch')
    plt.title('Classification vs Sparsity Loss')
    
    # 12. Mean gate value over epochs
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['mean_gate_l1'], label='Layer 1')
    plt.plot(epochs, history['mean_gate_l2'], label='Layer 2')
    plt.title('Mean Gate Value vs Epoch')
    plt.legend()
    
    # 14. Layer-wise sparsity vs epoch
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['l1_sparsity'], label='Layer 1')
    plt.plot(epochs, history['l2_sparsity'], label='Layer 2')
    plt.title('Layer-wise Sparsity vs Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/plots/experiment_curves_lambda_{lam}.png')
    plt.close()

def plot_gate_distributions(model, lam: float):
    """Plots histograms for final gate values ( overall + layer-wise)."""
    gates_all = []
    gates_layer = []
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                g = torch.sigmoid(module.gate_scores * 5).cpu().numpy().flatten()
                gates_all.extend(g)
                gates_layer.append(g)
                
    # 7. Overall Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(gates_all, bins=50, alpha=0.7)
    plt.title(f'Overall Gate Value Distribution (λ={lam})')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.savefig(f'results/plots/hist_overall_lambda_{lam}.png')
    plt.close()

    # 8. Layer-wise histograms
    fig, axes = plt.subplots(1, len(gates_layer), figsize=(12, 4))
    for i, g in enumerate(gates_layer):
        axes[i].hist(g, bins=50, alpha=0.7, color=f'C{i}')
        axes[i].set_title(f'Layer {i+1} Gate Distribution')
    plt.tight_layout()
    plt.savefig(f'results/plots/hist_layerwise_lambda_{lam}.png')
    plt.close()

def plot_cross_experiment_results(csv_path: str):
    """Plots comparing across all lambda values."""
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
        
    # 9. Accuracy vs Lambda
    plt.figure(figsize=(8, 5))
    plt.plot(df['Lambda'], df['Test Accuracy'], marker='o', label='Test Acc')
    plt.plot(df['Lambda'], df['Validation Accuracy'], marker='s', label='Val Acc')
    plt.xscale('log')
    plt.title('Accuracy vs Lambda')
    plt.xlabel('Lambda')
    plt.legend()
    plt.savefig('results/plots/acc_vs_lambda.png')
    plt.close()

    # 10. Sparsity vs Lambda
    plt.figure(figsize=(8, 5))
    plt.plot(df['Lambda'], df['Overall Sparsity'], marker='o')
    plt.xscale('log')
    plt.title('Overall Sparsity vs Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Sparsity %')
    plt.savefig('results/plots/sparsity_vs_lambda.png')
    plt.close()

    # 11. Accuracy vs Sparsity Trade-off
    plt.figure(figsize=(8, 5))
    plt.plot(df['Overall Sparsity'], df['Test Accuracy'], marker='o')
    plt.title('Accuracy vs Sparsity Trade-off')
    plt.xlabel('Sparsity %')
    plt.ylabel('Test Accuracy %')
    for i, lam in enumerate(df['Lambda']):
        plt.annotate(f'λ={lam}', (df['Overall Sparsity'].iloc[i], df['Test Accuracy'].iloc[i]))
    plt.savefig('results/plots/acc_vs_sparsity.png')
    plt.close()

    # 13. Layer-wise sparsity comparison bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df['Lambda']))
    width = 0.35
    plt.bar(x - width/2, df['Layer1 Sparsity'], width, label='Layer 1')
    plt.bar(x + width/2, df['Layer2 Sparsity'], width, label='Layer 2')
    plt.xticks(x, [f'λ={l}' for l in df['Lambda']])
    plt.ylabel('Sparsity %')
    plt.title('Layer-wise Sparsity cross-Lambda')
    plt.legend()
    plt.savefig('results/plots/layer_sparsity_bars.png')
    plt.close()

class PrunableLinear(nn.Module):
    """
    A custom linear layer that learns to prune its own weights during training.
    Contains standard weights, biases, and a set of learnable 'gate scores'.
    """
    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and biases
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Learnable gate scores, same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights using Kaiming normal.
        Initializes gate_scores using normal distribution (N(0, 0.01)) to start gates near 0.5.
        """
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Gate score initialization (Important for proper learning trajectory)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying dynamically scaled sigmoid gates to weights.
        """
        # Temperature scaling on gates to ensure sharper 0/1 states
        gates = torch.sigmoid(self.gate_scores * 5)
        
        # Apply pruning dynamically
        pruned_weights = self.weight * gates
        
        # Linear transformation
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningCNN(nn.Module):
    """
    A hybrid architecture featuring standard CNN feature extractors
    followed by the custom PrunableLinear layers for classification.
    """
    def __init__(self):
        super(SelfPruningCNN, self).__init__()
        # Standard CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Flattened size: 64 * 8 * 8 = 4096 (since CIFAR10 is 32x32 -> 16x16 -> 8x8)
        self.flatten = nn.Flatten()
        
        # Prunable fully connected layers
        self.fc1 = PrunableLinear(64 * 8 * 8, 256)
        self.relu = nn.ReLU()
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Extracts normalized L1 norm of all sigmoid gates across the network.
        Normalized by total gate count so lambda is architecture-independent.
        """
        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        total_gates = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores * 5)
                sparsity_loss += torch.sum(gates)
                total_gates += gates.numel()
        if total_gates > 0:
            sparsity_loss /= total_gates
        return sparsity_loss

    def get_layer_wise_stats(self, threshold: float = 1e-2) -> dict:
        """
        Calculates sparsity statistics per prunable layer.
        Returns:
            dict containing specific stats per layer index.
        """
        stats = {}
        layer_idx = 1
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                # We need to compute gates to calculate statistics
                with torch.no_grad():
                    gates = torch.sigmoid(module.gate_scores * 5)
                    total_weights = gates.numel()
                    
                    pruned_count = torch.sum(gates < threshold).item()
                    near_zero_count = torch.sum(gates < 1e-3).item()
                    near_one_count = torch.sum(gates > 0.9).item()
                    
                    stats[f'layer{layer_idx}_sparsity'] = (pruned_count / total_weights) * 100
                    stats[f'layer{layer_idx}_mean_gate'] = torch.mean(gates).item()
                    stats[f'layer{layer_idx}_near_zero'] = (near_zero_count / total_weights) * 100
                    stats[f'layer{layer_idx}_near_one'] = (near_one_count / total_weights) * 100
                    stats[f'layer{layer_idx}_weights'] = total_weights
                    
                layer_idx += 1
        return stats

    def get_overall_sparsity(self, threshold: float = 1e-2) -> tuple:
        """
        Returns overall sparsity % and mean gate value across the whole network.
        """
        total_weights = 0
        total_pruned = 0
        gate_sum = 0.0
        
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, PrunableLinear):
                    gates = torch.sigmoid(module.gate_scores * 5)
                    total_weights += gates.numel()
                    total_pruned += torch.sum(gates < threshold).item()
                    gate_sum += torch.sum(gates).item()
                    
        overall_sparsity = (total_pruned / total_weights) * 100 if total_weights > 0 else 0
        overall_mean_gate = gate_sum / total_weights if total_weights > 0 else 0
        
        return overall_sparsity, overall_mean_gate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_epoch(model, dataloader, criterion, optimizer, lam, device):
    model.train()
    running_loss, running_cls, running_spar = 0.0, 0.0, 0.0
    correct, total = 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        cls_loss = criterion(outputs, labels)
        spar_loss = model.get_sparsity_loss()
        
        loss = cls_loss + lam * spar_loss
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_cls += cls_loss.item() * inputs.size(0)
        running_spar += spar_loss.item() * inputs.size(0)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_cls = running_cls / total
    epoch_spar = running_spar / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_cls, epoch_spar, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            
            running_loss += cls_loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_experiment(lam, epochs, train_loader, val_loader, test_loader, device):
    logging.info(f"--- Starting experiment for Lambda = {lam} ---")
    model = SelfPruningCNN().to(device)
    
    # Split parameter groups to rescue gate scores from L2 weight decay
    base_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    
    optimizer = optim.Adam([
        {'params': base_params, 'weight_decay': 1e-4},
        {'params': gate_params, 'weight_decay': 0.0}
    ], lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'cls_loss', 'sparsity_loss', 
                               'overall_sparsity', 'mean_gate_l1', 'mean_gate_l2', 'l1_sparsity', 'l2_sparsity']}
    
    best_val_acc = 0.0
    pruning_onset_epoch = -1
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc=f"Training obj lam={lam}"):
        t_loss, t_cls, t_spar, t_acc = train_epoch(model, train_loader, criterion, optimizer, lam, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Get sparsity metrics
        overall_spar, _ = model.get_overall_sparsity()
        layer_stats = model.get_layer_wise_stats()
        
        # Pruning onset heuristic: record epoch when overall_sparsity crosses 5% rapidly
        if pruning_onset_epoch == -1 and overall_spar > 5.0:
            pruning_onset_epoch = epoch + 1
            
        # Logging history
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['cls_loss'].append(t_cls)
        history['sparsity_loss'].append(t_spar)
        history['overall_sparsity'].append(overall_spar)
        history['l1_sparsity'].append(layer_stats['layer1_sparsity'])
        history['l2_sparsity'].append(layer_stats['layer2_sparsity'])
        history['mean_gate_l1'].append(layer_stats['layer1_mean_gate'])
        history['mean_gate_l2'].append(layer_stats['layer2_mean_gate'])
        
        # Fail-Safe check at epoch 10
        if epoch + 1 == 10 and overall_spar < 5.0:
            logging.warning(f"[FAIL-SAFE] Sparsity is extremely low (<5%) after 10 epochs for lambda={lam}. Gate collapse might be failing.")
            
        # Checkpointing
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), f'results/best_model_lambda_{lam}.pth')
            
    train_time = time.time() - start_time
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(
        f'results/best_model_lambda_{lam}.pth',
        map_location=device,
        weights_only=True
    ))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    
    final_spar, final_mean_gate = model.get_overall_sparsity()
    final_stats = model.get_layer_wise_stats()
    
    # Generate per-experiment plots
    plot_experiment_curves(history, lam)
    plot_gate_distributions(model, lam)
    
    w1 = final_stats.get('layer1_weights', 1)
    w2 = final_stats.get('layer2_weights', 1)
    total_w = w1 + w2

    return {
        'Lambda': lam,
        'Test Accuracy': test_acc,
        'Validation Accuracy': best_val_acc,
        'Overall Sparsity': final_spar,
        'Overall Mean Gate': final_mean_gate,
        'Layer1 Sparsity': final_stats['layer1_sparsity'],
        'Layer2 Sparsity': final_stats['layer2_sparsity'],
        'Mean Gate Layer1': final_stats['layer1_mean_gate'],
        'Mean Gate Layer2': final_stats['layer2_mean_gate'],
        # Weighted average of layer-wise percentages based on number of weights
        '% Gates Near Zero': (final_stats.get('layer1_near_zero', 0) * w1 + final_stats.get('layer2_near_zero', 0) * w2) / total_w,
        '% Gates Near One': (final_stats.get('layer1_near_one', 0) * w1 + final_stats.get('layer2_near_one', 0) * w2) / total_w,
        'Pruning Onset Epoch': pruning_onset_epoch if pruning_onset_epoch != -1 else 'N/A',
        'Training Time': round(train_time, 2)
    }

def main():
    set_seed(42)
    prepare_results_dir()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
    lambdas = [0.001, 0.01, 0.1]
    epochs = 50
    
    all_results = []
    for lam in lambdas:
        res = train_experiment(lam, epochs, train_loader, val_loader, test_loader, device)
        all_results.append(res)
        
    csv_path = 'results/results_table.csv'
    fieldnames = ['Lambda', 'Test Accuracy', 'Validation Accuracy', 'Overall Sparsity', 'Overall Mean Gate',
                  'Layer1 Sparsity', 'Layer2 Sparsity', 'Mean Gate Layer1', 'Mean Gate Layer2',
                  '% Gates Near Zero', '% Gates Near One', 'Pruning Onset Epoch', 'Training Time']
                  
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
            
    plot_cross_experiment_results(csv_path)
    logging.info("Everything complete! Results saved in results/")

if __name__ == '__main__':
    main()
