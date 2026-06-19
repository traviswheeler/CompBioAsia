import gzip, time
import numpy as np
import torch
from torch import nn

with gzip.open('data.npy.gz', 'rb') as f:
    x = np.load(f)
x = x.reshape(-1, 700, 57)
aminos = torch.transpose(torch.tensor(x[:, :, 0:22]), -1, -2).float()
labels = torch.transpose(torch.tensor(x[:, :, 22:31]), -1, -2).float()
aminos_train, aminos_test = aminos[:5000], aminos[5000:]
labels_train, labels_test = labels[:5000], labels[5000:]

cross_entropy = nn.CrossEntropyLoss(ignore_index=8)
def loss_function(out, target_onehot):
    return cross_entropy(out, torch.argmax(target_onehot, dim=1))

def train_with_data(x, y, model, batch_size, steps, learning_rate, loss_function, checkin=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for step in range(steps):
        bi = torch.randint(0, len(x), (batch_size,))
        loss = loss_function(model(x[bi]), y[bi])
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        if step % checkin == 0:
            print('  progress: {:.0%}  loss: {:.4f}'.format(step / steps, loss.item()))

def test_accuracy(x, y, model, batch_size=32):
    correct = total = 0
    with torch.no_grad():
        for b in range(int(len(x) / batch_size) - 1):
            s = b * batch_size
            bx, by = x[s:s+batch_size], y[s:s+batch_size]
            call = torch.argmax(model(bx), dim=1)
            true = torch.argmax(by, dim=1)
            real = by[:, -1, :] == 0
            correct += torch.sum((call == true) & real)
            total += torch.sum(real)
    return float(correct / total)

def run(name, model, steps, lr):
    print(f"\n=== {name} ===")
    print("  params:", sum(p.numel() for p in model.parameters()))
    print("  before: %.4f" % test_accuracy(aminos_test, labels_test, model))
    t = time.time()
    train_with_data(aminos_train, labels_train, model, 32, steps, lr, loss_function, checkin=max(1, steps//4))
    print("  after:  %.4f   (%.0fs)" % (test_accuracy(aminos_test, labels_test, model), time.time()-t))

# ---- default notebook model ----
torch.manual_seed(0)
default = nn.Sequential(nn.Conv1d(22, 10, 1, padding='same'),
                        nn.ELU(),
                        nn.Conv1d(10, 9, 5, padding='same'))
run("DEFAULT (notebook starting point), 2000 steps, lr 1e-4", default, 2000, 1e-4)

# ---- a more appropriate, still-small model ----
# wider receptive field (kernel 11 over 3 conv layers) + more channels lets it
# use neighboring residues, which is what secondary structure actually depends on.
torch.manual_seed(0)
better = nn.Sequential(nn.Conv1d(22, 32, 11, padding='same'),
                       nn.ReLU(),
                       nn.Conv1d(32, 32, 11, padding='same'),
                       nn.ReLU(),
                       nn.Conv1d(32, 9, 11, padding='same'))
run("BETTER (3x conv, 32 ch, kernel 11), 3000 steps, lr 1e-3", better, 3000, 1e-3)
