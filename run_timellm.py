import pandas as pd
import os
import torch
from argparse import Namespace
from models.TimeLLM import Model

# Preprocess your data
df = pd.read_csv('normalized_training.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Separate numeric and non-numeric columns
data_numeric = df[['timestamp', 'TOTUSJH']].set_index('timestamp').resample('1H').mean()
data_flare = df[['timestamp', 'flare']].set_index('timestamp').resample('1H').first()
data = pd.concat([data_numeric, data_flare], axis=1).fillna(method='ffill')

# Ensure enough data
if len(data) < 196:
    raise ValueError(f"Dataset too short: {len(data)} rows. Need at least 196 rows.")

target_col = 'TOTUSJH'
if data['TOTUSJH'].std() == 0:
    print("TOTUSJH has no variance. Using USFLUX.")
    data_numeric = df[['timestamp', 'USFLUX']].set_index('timestamp').resample('1H').mean()
    data = pd.concat([data_numeric, data_flare], axis=1).fillna(method='ffill')
    target_col = 'USFLUX'

os.makedirs('./dataset_solar/solar', exist_ok=True)
train_data = data[[target_col]].iloc[:-100]
test_data = data[[target_col, 'flare']].iloc[-100:]
train_data.to_csv('./dataset_solar/solar/solar_flare_train.csv')
test_data.to_csv('./dataset_solar/solar/solar_flare_test.csv')
test_data[['flare']].to_csv('./dataset_solar/solar/solar_flare_test_labels.csv')

# Define configuration
configs = Namespace(
    task_name='long_term_forecast',
    seq_len=96,
    pred_len=24,
    label_len=48,
    enc_in=1,
    dec_in=1,
    c_out=1,
    d_model=512,
    d_ff=768,  # Match GPT-2’s llm_dim to fix shape mismatch
    llm_dim=768,
    llm_layers=12,
    n_heads=8,
    dropout=0.1,
    patch_len=16,
    stride=8,
    llm_model='GPT2',
    content='Solar flare forecasting based on physical parameters.',
    prompt_domain=True
)

# Initialize Time-LLM model
model = Model(configs)
model.eval()

# Prepare data for Time-LLM
train_tensor = torch.tensor(train_data[target_col].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
test_tensor = torch.tensor(test_data[target_col].values[-96:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
x_mark_enc = torch.zeros_like(test_tensor, dtype=torch.float32)
x_dec = torch.zeros((1, configs.pred_len + configs.label_len, 1), dtype=torch.float32)
x_mark_dec = torch.zeros_like(x_dec, dtype=torch.float32)

# Run Time-LLM inference with debugging
with torch.no_grad():
    print(f"test_tensor shape: {test_tensor.shape}")
    pred = model.forecast(test_tensor, x_mark_enc, x_dec, x_mark_dec)
    print(f"pred shape before squeeze: {pred.shape}")
    pred = pred.squeeze(0)
    print(f"pred shape after squeeze: {pred.shape}")

# Convert to numpy
pred_values = pred.numpy().flatten()
true_values = test_data[target_col].values[-24:]
labels = pd.read_csv('./dataset_solar/solar/solar_flare_test_labels.csv')
labels['flare_binary'] = labels['flare'].apply(lambda x: 0 if x == 'N' else 1)

threshold = 0.1  # Adjust based on your data
flare_pred = (pred_values > threshold).astype(int)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels['flare_binary'][-24:], flare_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize
import matplotlib.pyplot as plt
plt.plot(pred_values, label=f'Predicted {target_col}')
plt.plot(true_values, label=f'True {target_col}')
plt.plot(labels['flare_binary'][-24:], 'o', label='True Flare (1=Yes, 0=No)')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()