import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

MOD = 23
hidden = 64
test_sz = 0.5
l_rate = 0.0005
torch.manual_seed(42)

def binpow(a, n):
    global MOD
    if n == 0:
        return 1
    if n % 2 == 0:
        tmp = binpow(a, n / 2)
        return tmp * tmp % MOD
    return binpow(a, n - 1) * a % MOD

X = []
y = []

for i in range(MOD):
    for j in range(MOD):
        X.append([0] * i + [1] + [0] * (MOD - 1 - i) + [0] * j + [1] + [0] * (MOD - 1 - j))
        ans = binpow(j, MOD - 2) * i % MOD
        y.append([0] * ans + [1] + [0] * (MOD - 1 - ans))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state=42)

#X_train = X.copy()
#y_train = y.copy()

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

#print(X.shape, y.shape, X_train.shape, y_train.shape)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p = 0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        #self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = self.bn1(X)
        #X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.bn2(X)
        X = F.relu(self.fc3(X))
        X = self.bn3(X)
        X = F.softmax(self.fc4(X), dim=1)
        #X = self.dropout(X)
        return X

model = MLP(X_train.shape[1], hidden, MOD).to(device)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=0.01)

max_iter_count = 70000000

errors_mse = []
accuracy = []
accuracy_test = []
prob_train = []
weights_norms = []
gradient_norms = []
iter_num = []

for i in range(max_iter_count):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    L = loss(out, y_train)
    L.backward()
    optimizer.step()
    if i % (max_iter_count / 1000) == 0:
        model.eval()
        with torch.no_grad():
            errors_mse.append(L.item())
            predictions = out.argmax(dim=1)
            target = y_train.argmax(dim=1)
            accuracy.append((predictions == target).float().mean().item())

            out_test = model(X_test)
            predictions_test = out_test.argmax(dim=1)
            target_test = y_test.argmax(dim=1)
            accuracy_test.append((predictions_test == target_test).float().mean().item())
            iter_num.append(i)
            print(i / (max_iter_count / 100), '%', 'completed')
            print('loss:', L.item())
            print('accuracy_train:', accuracy[-1])
            print('accuracy_test:', accuracy_test[-1])

            total_norm = 0
            for p in model.parameters():
                total_norm += p.norm().item() ** 2
            weights_norms.append(total_norm ** 0.5)

            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            gradient_norms.append(total_grad_norm ** 0.5)

            prob_train.append(out.tolist())

df = pd.DataFrame({'errors_mse' : errors_mse, 'accuracy_train' : accuracy, 'accuracy_test' : accuracy_test, 'weights_norms' : weights_norms, 'gradient_norms' : gradient_norms, 'prob_train' : prob_train})
df.to_csv('very_important_file70kk.csv', index=False)


plt.figure(figsize=(6,4))
plt.plot(iter_num, errors_mse)
plt.xlabel('Optimization steps')
plt.ylabel('Cross-entropy loss')
plt.grid(True)
plt.savefig('loss_70kk.png')
plt.figure(figsize=(6,4))
plt.plot(iter_num, accuracy, label='train accuracy')
plt.plot(iter_num, accuracy_test, label='test accuracy')
plt.xlabel('Optimization steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_70kk.png')

torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model_backup70kk.pth')