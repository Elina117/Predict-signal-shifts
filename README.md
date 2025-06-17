# ⏱️ Predicting Temporal Shift in Sinusoidal Time Series with CNN-LSTM

The goal of this project is to develop a machine learning model for **regression of temporal shift** in synthetic time series. Each series is a sinusoidal signal with added noise and a random time shift. This task is relevant in domains such as audio analysis, biosignal processing, or synchronization of time stamps in data transmission systems.

## 🛠️ Technologies & Architecture

- **Language**: Python  
- **Framework**: PyTorch Lightning  
- **Model**: CNN + BiLSTM  
- **Data Generation**:
  - 15,000 signals, each of length 1500 time steps  
  - Random amplitude, frequency, and phase  
  - Gaussian noise added  
  - Random shift from -10 to +10 time steps  
- **Preprocessing**:
  - Signal-wise normalization (zero mean, unit std)  
  - Train/validation split: 80/20  
- **Model Architecture**:
  - `Conv1d (in=1, out=32, kernel=5, padding=3)` → `BatchNorm` → `ReLU`
  - `Conv1d (32→64)` → `BatchNorm` → `ReLU` → `MaxPool1d`
  - `BiLSTM (input_size=64, hidden_size=64, num_layers=2, dropout=0.3)`
  - Fully connected block: `Linear → ReLU → Dropout → Linear → Output`
- **Training**:
  - Optimizer: Adam (lr=1e-3)  
  - Scheduler: ReduceLROnPlateau  
  - Loss function: MSE  
  - Epochs: 30  
  - Batch size: 32  

## 📊 Results

| Metric                    | Value            |
|---------------------------|------------------|
| Dataset size              | 15,000 signals   |
| Signal length             | 1500 time steps  |
| Best validation MSE       | **~7.8**         |
| Training epochs           | 30               |
| Model stability           | High             |

The model demonstrated stable and consistent training across multiple configurations of the input data. 🚀
