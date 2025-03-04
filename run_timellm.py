import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from argparse import Namespace
from models.TimeLLM import Model

# Load datasets
train_df = pd.read_csv('normalized_training.csv')
val_df = pd.read_csv('normalized_validation.csv')
test_df = pd.read_csv('normalized_testing.csv')

# Convert timestamps
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

# Numeric features
numeric_cols = [col for col in train_df.columns if col not in ['timestamp', 'NOAA', 'HARP', 'label', 'flare']]
print(f"Using {len(numeric_cols)} features: {numeric_cols}")

# Normalize features (remove if already normalized)
scalers = {col: StandardScaler() for col in numeric_cols}
for col in numeric_cols:
    train_df[col] = scalers[col].fit_transform(train_df[[col]])
    val_df[col] = scalers[col].transform(val_df[[col]])
    test_df[col] = scalers[col].transform(test_df[[col]])

# Prepare binary flare labels
train_df['flare_binary'] = train_df['flare'].apply(lambda x: 0 if pd.isna(x) or x == 'N' else 1)
val_df['flare_binary'] = val_df['flare'].apply(lambda x: 0 if pd.isna(x) or x == 'N' else 1)
test_df['flare_binary'] = test_df['flare'].apply(lambda x: 0 if pd.isna(x) or x == 'N' else 1)

# Prepare data
train_features = train_df[numeric_cols]
train_labels = train_df['flare_binary']
val_features = val_df[numeric_cols]
val_labels = val_df['flare_binary']
test_features = test_df[numeric_cols]
test_labels = test_df['flare_binary']

# Config (revert task_name and adjust c_out)
configs = Namespace(
    task_name='long_term_forecast',  # Match original TimeLLM expectation
    seq_len=48, pred_len=12, label_len=24,
    enc_in=len(numeric_cols), dec_in=len(numeric_cols), c_out=len(numeric_cols),  # Output all features
    d_model=256, d_ff=512, llm_dim=768, llm_layers=12, n_heads=4,
    dropout=0.3, patch_len=8, stride=4,
    llm_model='GPT2', content='Solar flare classification with multiple physical parameters.', prompt_domain=True
)

# Prepare sequences
def prepare_sequences(features, labels, seq_len, pred_len):
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len + 1):
        X.append(features.iloc[i:i+seq_len].values)
        y.append(labels.iloc[i+seq_len:i+seq_len+pred_len].values.max())  # 1 if any flare in pred_len
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

train_X, train_y = prepare_sequences(train_features, train_labels, configs.seq_len, configs.pred_len)
val_X, val_y = prepare_sequences(val_features, val_labels, configs.seq_len, configs.pred_len)
test_X, test_y = prepare_sequences(test_features, test_labels, configs.seq_len, configs.pred_len)

# Custom classifier
class TimeLLMClassifier(nn.Module):
    def __init__(self, configs):
        super(TimeLLMClassifier, self).__init__()
        self.time_llm = Model(configs)
        self.classifier = nn.Sequential(
            nn.Linear(configs.c_out * configs.pred_len, 128),  # Flatten pred_len features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Single binary output
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        time_llm_out = self.time_llm(x_enc, x_mark_enc, x_dec, x_mark_dec)  # Shape: (batch, pred_len, c_out)
        batch_size = time_llm_out.shape[0]
        flat_out = time_llm_out.reshape(batch_size, -1)  # Shape: (batch, pred_len * c_out)
        class_out = self.classifier(flat_out)  # Shape: (batch, 1)
        return class_out

# Model and training
model = TimeLLMClassifier(configs)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

# Training loop
epochs = 50
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    x_enc = train_X
    x_dec = torch.zeros((train_X.shape[0], configs.pred_len, configs.enc_in), dtype=torch.float32)
    x_mark_enc = torch.zeros_like(x_enc)
    x_mark_dec = torch.zeros_like(x_dec)

    logits = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # Shape: (batch, 1)
    loss = criterion(logits, train_y.unsqueeze(1))  # Match shapes
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X, torch.zeros_like(val_X),
                          torch.zeros((val_X.shape[0], configs.pred_len, configs.enc_in)),
                          torch.zeros((val_X.shape[0], configs.pred_len, configs.enc_in)))
        val_loss = criterion(val_logits, val_y.unsqueeze(1))

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Inference
with torch.no_grad():
    test_logits = model(test_X, torch.zeros_like(test_X),
                        torch.zeros((test_X.shape[0], configs.pred_len, configs.enc_in)),
                        torch.zeros((test_X.shape[0], configs.pred_len, configs.enc_in)))
    test_probs = torch.sigmoid(test_logits).numpy()  # Shape: (batch, 1)
    test_preds = (test_probs > 0.5).astype(int).flatten()

# TSS Calculation
true_labels = test_y.numpy()
TP = np.sum((test_preds == 1) & (true_labels == 1))
TN = np.sum((test_preds == 0) & (true_labels == 0))
FP = np.sum((test_preds == 1) & (true_labels == 0))
FN = np.sum((test_preds == 0) & (true_labels == 1))
tss = (TP / (TP + FN)) - (FP / (FP + TN)) if (TP + FN > 0 and FP + TN > 0) else 0
print(f"Test TSS Score: {tss:.4f}")

# Visualization
plt.plot(test_probs, label='Predicted Flare Probability')
plt.plot(true_labels, 'o', label='True Flare (1=Yes, 0=No)')
plt.axhline(0.5, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()