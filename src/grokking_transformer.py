import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LinearLR, LambdaLR
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class MultiHeadAttention(nn.Module):
    def __init__(self, dm, head_count):
        super().__init__()
        assert dm % head_count == 0
        self.head_count = head_count
        self.dh = dm // head_count
        self.qkv = nn.Linear(dm, 3 * dm)
        self.out = nn.Linear(dm, dm)
    
    def forward(self, X):
        b, t, d = X.size()
        qkv = self.qkv(X).reshape(b, t, self.head_count, 3 * self.dh)
        q, k, v = qkv.chunk(3, dim=-1) # b * t * self.head_count * self.dh
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # b * self.head_count * t * self.dh
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) # b * self.head_count * t * t
        att = F.softmax(att, dim=-1)
        out = att @ v # b * self.head_count * t * self.dh
        out = out.transpose(1, 2)
        out = out.reshape(b, t, d)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dm, head_count):
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, head_count)
        self.ln1 = nn.LayerNorm(dm)
        self.mlp1 = nn.Sequential(nn.Linear(dm, 4 * dm), nn.ReLU(), nn.Linear(4 * dm, dm))
        self.ln2 = nn.LayerNorm(dm)
    
    def forward(self, X):
        out = self.mha1(X) + X
        out = self.ln1(out)
        out = self.mlp1(out) + out
        out = self.ln2(out)
        return out

class Transformer(nn.Module):
    def __init__(self, vocab_size, dm = 128, head_count = 4, layer_count = 2, seq_len = 2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dm)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, dm))
        self.layers = nn.ModuleList([TransformerBlock(dm, head_count) for _ in range(layer_count)])
        self.ln1 = nn.LayerNorm(dm)
        self.fc1 = nn.Linear(dm, vocab_size)
    
    def forward(self, X):
        b, t = X.size()
        X = self.token_emb(X) + self.pos_emb[:,:t]
        for layer in self.layers:
            X = layer(X)
        X = self.ln1(X)
        return self.fc1(X)

MOD = 97
test_sz = 0.5
l_rate = 0.001
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
        X.append([i, j])
        ans = binpow(j, MOD - 2) * i % MOD
        y.append(ans)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz, random_state=42)

X = torch.tensor(X, dtype=torch.long).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)
X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

model = Transformer(MOD).to(device)

loss = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=0.01, betas=torch.tensor([0.9, 0.98]), capturable=True, foreach=True)

max_iter_count = 1000000

def linear_warmup(i):
    if i < 10:
        return i / 10
    return 1.0

sheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)


errors = []
accuracy = []
accuracy_test = []
iter_num = []

for i in range(max_iter_count):
    model.train()
    total_L = 0
    for samples, answers in train_loader:
        samples = samples.to(device)
        answers = answers.to(device)
        optimizer.zero_grad()
        out = model(X_train)[:,-1,:]
        L = loss(out, y_train)
        total_L += L.item()
        L.backward()
        optimizer.step()
        sheduler.step()
    if i % (max_iter_count / 100) == 0:
        iter_num.append(i)
        model.eval()
        correct_train = 0
        correct_test = 0
        total_train = 0
        total_test = 0
        with torch.no_grad():
            predictions = out.argmax(dim=1)
            target = y_train
            accuracy.append((predictions == target).float().mean().item())

            out_test = model(X_test)[:,-1,:]
            predictions_test = out_test.argmax(dim=1)
            target_test = y_test
            accuracy_test.append((predictions_test == target_test).float().mean().item())
            errors.append(total_L)
            print(i / (max_iter_count / 100), '%', 'completed')
            print('loss:', total_L)
            print('accuracy_train:', accuracy[-1])
            print('accuracy_test:', accuracy_test[-1])

df = pd.DataFrame({'errors' : errors, 'accuracy_train' : accuracy, 'accuracy_test' : accuracy_test})
df.to_csv('stats_grokking_transformer.csv', index=False)


plt.figure(figsize=(6,4))
plt.plot(iter_num, errors)
plt.xlabel('Optimization steps')
plt.ylabel('Cross-entropy loss')
plt.grid(True)
plt.savefig('loss_transformer.png')
plt.figure(figsize=(6,4))
plt.plot(iter_num, accuracy, label='train accuracy')
plt.plot(iter_num, accuracy_test, label='test accuracy')
plt.xlabel('Optimization steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_transformer.png')

torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model_backup_transformer.pth')