import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

# ============================== [0] 경로 설정 ==============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

# ============================== [1] 데이터셋 클래스 ==============================
class StockDataset(Dataset):
    def __init__(self, csv_path, seq_len=60, pred_horizon=[1, 3, 5]):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df['stk_cd'] = df['stk_cd'].astype(str).str.zfill(6)
        
        # 0. 데이터 클렌징 (NaN/Inf 제거)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.raw_df = df.copy()
        
        # 1. 피처와 타겟 분리
        feature_df = df.drop(columns=['dt', 'stk_cd'])
        target_idx = feature_df.columns.get_loc('close')
        
        # 2. 전체 피처 스케일링 (중요: NaN 방지의 핵심)
        self.full_scaler = StandardScaler()
        scaled_data = self.full_scaler.fit_transform(feature_df)
        
        # 별도의 가격 역산용 스케일러 저장
        self.price_scaler = StandardScaler()
        self.price_scaler.fit(feature_df.values[:, target_idx].reshape(-1, 1))
        
        self.X, self.y = [], []
        max_h = max(pred_horizon)
        
        # 종목별 시퀀스 생성
        for code in df['stk_cd'].unique():
            # 스케일링된 데이터에서 해당 종목 필터링
            mask = (df['stk_cd'] == code).values
            stock_data = scaled_data[mask]
            
            if len(stock_data) < (seq_len + max_h): continue
            
            for i in range(len(stock_data) - seq_len - max_h + 1):
                self.X.append(stock_data[i : i + seq_len])
                # 미래 시점 종가 (이미 스케일링된 값 사용)
                targets = [stock_data[i + seq_len + h - 1, target_idx] for h in pred_horizon]
                self.y.append(targets)
        
        self.X = np.array(self.X).astype(np.float32)
        self.y = np.array(self.y).astype(np.float32)
        print(f"✅ 데이터셋 생성 완료: {len(self.X)} 샘플 | 피처 수: {feature_df.shape[1]}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): 
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# ============================== [2] Transformer 모델 ==============================
class StockTransformer(nn.Module):
    def __init__(self, feature_dim, seq_len=60, num_predictions=3):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, 128)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, 128) * 0.01)
        
        # batch_first=True 설정 시 (batch, seq, feature) 순서
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, 
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_predictions)
        )
        
    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding
        x = self.encoder(x)
        x = x.mean(dim=1) 
        return self.fc(x)

# ============================== [3] 메인 실행부 ==============================
if __name__ == "__main__":
    CSV_PATH = DATA_DIR / "total_top10_train_data.csv" # final_merge.py가 만든 파일명으로 확인 필요
    SEQ_LEN = 60
    PRED_HORIZON = [1, 3, 5]
    
    ds = StockDataset(CSV_PATH, seq_len=SEQ_LEN, pred_horizon=PRED_HORIZON)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    model = StockTransformer(ds.X.shape[2], seq_len=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) # lr 살짝 낮춤
    criterion = nn.MSELoss()
    
    print(f"🚀 학습 시작... (장치: {DEVICE})")
    model.train()
    for epoch in range(200):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            
            if torch.isnan(loss):
                print("⚠️ Loss가 NaN입니다. 학습을 중단합니다.")
                break
                
            loss.backward()
            # Gradient Clipping 추가 (안정성 강화)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(loader):.6f}")

    # ============================== [4] 예측 및 시각화 ==============================
    TARGET_CODE = "000660" 
    model.eval()
    with torch.no_grad():
        target_df = ds.raw_df[ds.raw_df['stk_cd'] == TARGET_CODE]
        if not target_df.empty:
            feature_cols = [c for c in target_df.columns if c not in ['dt', 'stk_cd']]
            # 전체 스케일러 적용 필요
            last_raw = target_df[feature_cols].values[-SEQ_LEN:]
            last_seq = ds.full_scaler.transform(last_raw).astype(np.float32)
            last_seq_tensor = torch.from_numpy(last_seq).unsqueeze(0).to(DEVICE)
            
            future_scaled = model(last_seq_tensor)
            future_prices = ds.price_scaler.inverse_transform(future_scaled.cpu().numpy())
            
            print("\n" + "="*55)
            print(f"🔮 종목코드 [{TARGET_CODE}] 미래 주가 예측 결과")
            for i, h in enumerate(PRED_HORIZON):
                print(f"  ▶️ {h}일 후 예상 종가: {future_prices[0, i]:,.0f} 원")
            print("="*55 + "\n")

    # 시각화 및 저장
    with torch.no_grad():
        X_test = torch.from_numpy(ds.X[-300:]).to(DEVICE)
        pred_scaled = model(X_test).cpu().numpy()
        actual_scaled = ds.y[-300:]
        
        # 역정규화 (가격 단위 복원)
        pred_final = ds.price_scaler.inverse_transform(pred_scaled)
        actual_final = ds.price_scaler.inverse_transform(actual_scaled)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for i, h in enumerate(PRED_HORIZON):
        axes[i].plot(actual_final[:, i], label='Actual', color='blue', alpha=0.6)
        axes[i].plot(pred_final[:, i], label='Predicted', color='red', linestyle='--')
        axes[i].set_title(f"Forecast: +{h} Day(s)")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'final_prediction_plot.png')
    torch.save(model.state_dict(), DATA_DIR / "transformer_model.pth")
    joblib.dump(ds.full_scaler, DATA_DIR / "full_scaler.pkl")
    joblib.dump(ds.price_scaler, DATA_DIR / "price_scaler.pkl")
    print(f"💾 결과 저장 완료 (위치: {DATA_DIR})")
    plt.show()