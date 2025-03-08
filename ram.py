import pandas as pd
import torch
import warnings
from argparse import Namespace
from models.TimeLLM import Model
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_csv('normalized_training.csv')
val_df = pd.read_csv('normalized_validation.csv')
test_df = pd.read_csv('normalized_testing.csv')

# Preprocess data
def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = ['flare']
    df_numeric = df.set_index('timestamp')[numeric_cols].resample('1H').mean()
    df_categorical = df.set_index('timestamp')[categorical_cols].resample('1H').first()
    df_processed = pd.concat([df_numeric, df_categorical], axis=1).fillna(method='ffill')
    return df_processed

train_data = preprocess_data(train_df)
val_data = preprocess_data(val_df)
test_data = preprocess_data(test_df)

# Ensure enough data
min_length = 196
for data, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
    if len(data) < min_length:
        raise ValueError(f"{name} dataset too short: {len(data)} rows. Need at least {min_length} rows.")

# Feature columns
feature_cols = [col for col in train_data.columns if col != 'flare' and train_data[col].dtype in ['float64', 'int64']]
target_col = 'flare'

# Convert flare to binary
for data in [train_data, val_data, test_data]:
    data['flare_binary'] = data['flare'].apply(lambda x: 0 if x == 'N' else 1)

# Calculate class weights for imbalance
pos_weight = train_data['flare_binary'].value_counts()[0] / train_data['flare_binary'].value_counts()[1] if 1 in train_data['flare_binary'].values else 1.0
pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

# Define configuration
configs = Namespace(
    task_name='long_term_forecast',
    seq_len=96,
    pred_len=24,
    label_len=48,
    enc_in=len(feature_cols),
    dec_in=len(feature_cols),
    c_out=1,  # Ensure this is respected by the model
    d_model=512,
    d_ff=768,
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

# Initialize model
model = Model(configs)

# Prepare tensors
def prepare_tensors(data, feature_cols, seq_len, pred_len, label_len):
    features = data[feature_cols].values
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    x_enc = tensor[:, -seq_len:, :]
    x_mark_enc = torch.zeros_like(x_enc, dtype=torch.float32)
    x_dec = torch.zeros((1, pred_len + label_len, len(feature_cols)), dtype=torch.float32)
    x_mark_dec = torch.zeros_like(x_dec, dtype=torch.float32)
    return x_enc, x_mark_enc, x_dec, x_mark_dec

# Calculate TSS
def calculate_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tss = (tp / (tp + fn)) - (fp / (fp + tn)) if (tp + fn) > 0 and (fp + tn) > 0 else 0
    return tss

# Training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Add class weight
best_val_loss = float('inf')
patience, max_epochs = 5, 50  # Early stopping

for epoch in range(max_epochs):
    x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_tensors(train_data, feature_cols, configs.seq_len, configs.pred_len, configs.label_len)
    target = torch.tensor(train_data['flare_binary'].values[-configs.pred_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    optimizer.zero_grad()
    pred = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    if pred.shape[-1] != 1:
        pred = pred[..., :1]  # Adjust if necessary
    
    # Debug: Print shapes and values
    if epoch == 0:
        print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
        print(f"Pred sample: {pred[0, :5, 0]}, Target sample: {target[0, :5, 0]}")
    
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Validation for early stopping
    with torch.no_grad():
        val_x_enc, val_x_mark_enc, val_x_dec, val_x_mark_dec = prepare_tensors(val_data, feature_cols, configs.seq_len, configs.pred_len, configs.label_len)
        val_target = torch.tensor(val_data['flare_binary'].values[-configs.pred_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        val_pred = model.forecast(val_x_enc, val_x_mark_enc, val_x_dec, val_x_mark_dec)
        if val_pred.shape[-1] != 1:
            val_pred = val_pred[..., :1]
        val_loss = criterion(val_pred, val_target)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Validation
model.eval()
with torch.no_grad():
    x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_tensors(val_data, feature_cols, configs.seq_len, configs.pred_len, configs.label_len)
    val_pred = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    if val_pred.shape[-1] != 1:
        val_pred = val_pred[..., :1]
    val_pred_binary = (torch.sigmoid(val_pred) > 0.5).float().squeeze(0).numpy()
    val_true = val_data['flare_binary'].values[-configs.pred_len:]
    val_accuracy = accuracy_score(val_true, val_pred_binary)
    val_tss = calculate_tss(val_true, val_pred_binary)
    print(f"Validation Accuracy: {val_accuracy:.2f}, TSS: {val_tss:.2f}")

# Testing
with torch.no_grad():
    x_enc, x_mark_enc, x_dec, x_mark_dec = prepare_tensors(test_data, feature_cols, configs.seq_len, configs.pred_len, configs.label_len)
    test_pred = model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    if test_pred.shape[-1] != 1:
        test_pred = test_pred[..., :1]
    test_pred_values = torch.sigmoid(test_pred).squeeze(0).numpy()
    test_pred_binary = (test_pred_values > 0.5).astype(int)
    test_true = test_data['flare_binary'].values[-configs.pred_len:]
    test_accuracy = accuracy_score(test_true, test_pred_binary)
    test_tss = calculate_tss(test_true, test_pred_binary)
    print(f"Test Accuracy: {test_accuracy:.2f}, TSS: {test_tss:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(test_pred_values, label='Predicted Flare Probability')
plt.plot(test_true, 'o', label='True Flare (1=Yes, 0=No)')
plt.axhline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
plt.title(f'Solar Flare Prediction (Test TSS: {test_tss:.2f})')
plt.xlabel('Time Step')
plt.ylabel('Probability / Binary')
plt.legend()
plt.show()