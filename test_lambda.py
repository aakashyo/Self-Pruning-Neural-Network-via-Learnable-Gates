import solution
from solution import get_dataloaders, train_epoch, SelfPruningCNN
import torch
import torch.nn as nn
import torch.optim as optim

solution.set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _, _ = get_dataloaders(batch_size=128)

model = SelfPruningCNN().to(device)
base_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]

optimizer = optim.Adam([
    {'params': base_params, 'weight_decay': 1e-4},
    {'params': gate_params, 'weight_decay': 0.0}
], lr=1e-3)
criterion = nn.CrossEntropyLoss()

lam = 0.001
for epoch in range(2):
    t_loss, t_cls, t_spar, t_acc = train_epoch(model, train_loader, criterion, optimizer, lam, device)
    print(f'Epoch {epoch}: Sparsity Loss={t_spar}, Acc={t_acc}')
    spar, gate = model.get_overall_sparsity()
    print(f'Overall Sparsity: {spar}, Mean Gate: {gate}')
