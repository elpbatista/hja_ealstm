import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Settings ===
SEQUENCE_LENGTH = 270
BATCH_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load data ===
X = np.load("data/processed/x_sequences.npy")
y = np.load("data/processed/y_targets.npy").reshape(-1, 1)

# === Normalize inputs and outputs ===
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_flat = X.reshape(-1, X.shape[2])
X_scaled = x_scaler.fit_transform(X_flat).reshape(X.shape)
y_scaled = y_scaler.fit_transform(y)

# Save scalers for later use (optional)
Path("models").mkdir(parents=True, exist_ok=True)
np.save("models/x_mean.npy", x_scaler.mean_)
np.save("models/x_std.npy", x_scaler.scale_)
np.save("models/y_mean.npy", y_scaler.mean_)
np.save("models/y_std.npy", y_scaler.scale_)

# Split into train and test
split = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# === Define LSTM Model ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = LSTMRegressor(input_size=X.shape[2], hidden_size=HIDDEN_SIZE).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# === NSE metric ===
def nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

# === Training Loop with Early Stopping and LR Decay ===
best_loss = float("inf")
epochs_no_improve = 0
best_model_path = Path("models/lstm_best_model.pt")

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor.to(DEVICE)).cpu().squeeze()
        val_loss = loss_fn(val_preds, y_test_tensor.squeeze()).item()
        val_nse = nse(y_test_tensor.squeeze().numpy(), val_preds.numpy())
        scheduler.step(val_loss)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} — Val MSE: {val_loss:.4f} — NSE: {val_nse:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# === Final Evaluation ===
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    final_preds = model(X_test_tensor.to(DEVICE)).cpu().numpy()

# === Denormalize
y_true = y_scaler.inverse_transform(y_test_tensor.numpy())
y_pred = y_scaler.inverse_transform(final_preds)

Path("results").mkdir(parents=True, exist_ok=True)
np.save("results/y_true.npy", y_true)
np.save("results/y_pred.npy", y_pred)

# === Plots
plt.figure(figsize=(10, 5))
plt.plot(y_true, label="Observed")
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("LSTM Discharge Prediction (Denormalized)")
plt.xlabel("Day")
plt.ylabel("Discharge (mm/day)")
plt.tight_layout()
plt.savefig("results/lstm_prediction_normalized.png")
plt.show()

# Residuals
residuals = y_true.squeeze() - y_pred.squeeze()
plt.figure(figsize=(10, 4))
plt.plot(residuals, label="Residuals", color="orange")
plt.axhline(0, linestyle="--", color="black", linewidth=1)
plt.title("Prediction Residuals (Observed - Predicted)")
plt.xlabel("Day")
plt.ylabel("Residual (mm/day)")
plt.tight_layout()
plt.savefig("results/lstm_residuals_normalized.png")
plt.show()
