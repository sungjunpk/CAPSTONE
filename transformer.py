import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# =========================
# 경로 & 디바이스
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() 
                     else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 현재 사용 중인 연산 장치: {DEVICE}")

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# =========================
# Dataset
# =========================
class StockDataset(Dataset):
    def __init__(self, csv_path, seq_len=90, pred_horizon=[1, 3, 5], 
                 train=True, train_ratio=0.8, scalers=None):
        df = pd.read_csv(csv_path)
        df['stk_cd'] = df['stk_cd'].astype(str).str.zfill(6)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        drop_cols = ['dt', 'stk_cd', 'stk_name']
        self.feature_cols = [c for c in df.columns if c not in drop_cols]
        
        self.X, self.y = [], []
        max_h = max(pred_horizon)
        self.scalers = scalers if scalers is not None else {}
        
        for code in df['stk_cd'].unique():
            mask = (df['stk_cd'] == code).values
            stock_data = df[self.feature_cols].values[mask]
            
            if len(stock_data) < seq_len + max_h + 30:
                continue
                
            split_idx = int(len(stock_data) * train_ratio)
            data = stock_data[:split_idx] if train else stock_data[split_idx:]
            
            if train:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                self.scalers[code] = scaler
            else:
                if code not in self.scalers:
                    continue
                scaled_data = self.scalers[code].transform(data)
            
            # Target: log return
            price_idx = self.feature_cols.index('close')
            prices = stock_data[:, price_idx]
            log_returns = np.log(prices / np.roll(prices, 1))
            log_returns[0] = 0.0
            
            for i in range(len(data) - seq_len - max_h + 1):
                self.X.append(scaled_data[i:i + seq_len])
                targets = [log_returns[i + seq_len + h - 1] for h in pred_horizon]
                self.y.append(targets)
        
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
        mode = "Train" if train else "Validation"
        print(f"✅ {mode} 데이터셋: {len(self.X):,} 샘플 | seq_len={seq_len} | 피처: {len(self.feature_cols)}개")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# =========================
# Transformer 모델
# =========================
class StockTransformer(nn.Module):
    def __init__(self, feature_dim: int, seq_len: int = 90, num_predictions: int = 3):
        super().__init__()
        d_model = 80          # overfitting 줄이기 위해 더 낮춤
        nhead = 8
        num_layers = 3        # 4 → 3으로 더 가볍게
        
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 150)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=192,
            dropout=0.3,            # dropout 강화
            activation='gelu',
            batch_first=True, 
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_predictions)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x)

# =========================
# Loss
# =========================
class HuberLoss(nn.Module):
    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        return nn.HuberLoss(delta=self.delta)(pred, target)

# =========================
# 메인
# =========================
if __name__ == "__main__":
    CSV_PATH = DATA_DIR / "capstone_semiconductor_train_data_fixed.csv"
    
    SEQ_LEN = 90                    # ← 60 → 90으로 증가 (추천)
    PRED_HORIZON = [1, 3, 5]
    BATCH_SIZE = 64
    PATIENCE = 10
    MAX_EPOCHS = 150
    
    # Train / Validation Dataset
    train_ds = StockDataset(
        CSV_PATH, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON, 
        train=True, train_ratio=0.8
    )
    
    val_ds = StockDataset(
        CSV_PATH, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON, 
        train=False, train_ratio=0.8, scalers=train_ds.scalers
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    model = StockTransformer(
        feature_dim=train_ds.X.shape[2],
        seq_len=SEQ_LEN,
        num_predictions=len(PRED_HORIZON)
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = HuberLoss(delta=0.5)
    
    # verbose 제거 (PyTorch 최신 버전)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = DATA_DIR / "best_transformer_v2.pth"
    
    print("\n🚀 Transformer 학습 시작...\n")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   → Best Model Saved! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("🛑 Early Stopping")
                break
    
    print(f"\n🎉 학습 완료! Best Val Loss: {best_val_loss:.6f}")
    print(f"💾 Best 모델 저장: {best_model_path}")