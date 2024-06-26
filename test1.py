import optuna
import yaml
import torch
import torch.optim as optim
from torchvision import datasets, transforms

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

def validate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetBackbone(num_classes=4).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.FakeData(transform=transforms.ToTensor()), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FakeData(transform=transforms.ToTensor()), batch_size=1000, shuffle=False)
    
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
    
    accuracy = validate(model, device, test_loader)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Save best hyperparameters to YAML file
best_params = trial.params
with open('best_hyperparameters.yaml', 'w') as f:
    yaml.dump(best_params, f)
