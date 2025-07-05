import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Settings ===
SEQUENCE_LENGTH = 270
BATCH_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load data ===
X = np.load("data/processed/x_sequences.npy")
y = np.load("data/processed/y_targets.npy")

# Split into train and test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# === Define LSTM Model ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

model = LSTMRegressor(input_size=X.shape[2], hidden_size=HIDDEN_SIZE).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor.to(DEVICE)).cpu().squeeze()
        val_loss = loss_fn(val_preds, y_test_tensor.squeeze()).item()
    print(f"Epoch {epoch+1:2d}/{EPOCHS} — Val MSE: {val_loss:.4f}")

# === Save outputs
Path("results").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Save model weights
torch.save(model.state_dict(), "models/lstm_model.pt")

# Save predictions and true values
y_true = y_test_tensor.squeeze().numpy()
y_pred = val_preds.numpy()
np.save("results/y_true.npy", y_true)
np.save("results/y_pred.npy", y_pred)

# Save plot
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="Observed")
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("LSTM Discharge Prediction")
plt.xlabel("Day")
plt.ylabel("Discharge (mm/day)")
plt.tight_layout()
plt.savefig("results/lstm_prediction.png")
plt.show()
