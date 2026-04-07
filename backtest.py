import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ====================== 설정 ======================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "csv"
MODEL_PATH = DATA_DIR / "best_transformer.pth"
CSV_PATH = DATA_DIR / "capstone_semiconductor_train_data_fixed.csv"

SEQ_LEN = 30
DEVICE = torch.device("cpu")
THRESHOLD = 0.0003   # 거래 빈도 조절 (0.0001 ~ 0.0005 추천)

# ====================== 모델 정의 ======================
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockTransformer(torch.nn.Module):
    def __init__(self, feature_dim: int, seq_len: int = 30, num_predictions: int = 3):
        super().__init__()
        d_model = 128
        self.input_proj = torch.nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 50)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=0.15, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, num_predictions)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.fc(x)

# ====================== 데이터 & 모델 로드 ======================
df = pd.read_csv(CSV_PATH)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

drop_cols = ['dt', 'stk_cd', 'stk_name']
feature_cols = [c for c in df.columns if c not in drop_cols]
feature_dim = len(feature_cols)

model = StockTransformer(feature_dim=feature_dim, seq_len=SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ 모델 및 데이터 로드 완료\n")

# ====================== 개선된 백테스트 ======================
def run_improved_backtest():
    total_preds = []
    total_actuals = []
    results = []
    
    print(f"{'종목':12s} {'방향정확도':>10} {'수익률':>8} {'거래횟수':>8} {'MAE(%)':>8} {'RMSE(%)':>8}")
    print("-" * 75)
    
    for code in sorted(df['stk_cd'].unique()):
        stock_df = df[df['stk_cd'] == code].copy().reset_index(drop=True)
        stock_name = stock_df['stk_name'].iloc[0]
        
        if len(stock_df) < SEQ_LEN + 120:
            continue
            
        # Temporal Split
        split_idx = int(len(stock_df) * 0.8)
        train_df = stock_df.iloc[:split_idx]
        test_df = stock_df.iloc[split_idx:].reset_index(drop=True)
        
        # === Train 기간으로 Scaler 학습 (중요!) ===
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols])
        
        preds = []
        actuals = []
        capital = 100_000_000.0
        position = 0.0
        trades = 0
        holding_days = 0
        
        for i in range(SEQ_LEN, len(test_df)):
            seq = test_df.iloc[i-SEQ_LEN:i][feature_cols].values
            scaled_seq = scaler.transform(seq)                    # Train scaler 사용!
            scaled_seq = np.nan_to_num(scaled_seq)
            
            input_tensor = torch.FloatTensor(scaled_seq).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                pred_return = model(input_tensor)[0, 0].item()
            
            actual_return = test_df.iloc[i]['log_return']
            
            preds.append(pred_return)
            actuals.append(actual_return)
            
            current_price = test_df.iloc[i]['close']
            holding_days += 1 if position > 0 else 0
            
            # 매매 로직
            if position == 0 and pred_return > THRESHOLD:
                position = capital * 0.98 / current_price
                capital = 0
                trades += 1
                holding_days = 0
            elif position > 0 and (pred_return < -THRESHOLD or holding_days >= 8):
                capital = position * current_price * 0.999
                position = 0
                trades += 1
                holding_days = 0
            
            if position > 0:
                capital = position * current_price
        
        if position > 0:
            capital = position * test_df.iloc[-1]['close']
        
        total_return = (capital / 100_000_000 - 1) * 100
        
        # 정확도 계산
        preds_arr = np.array(preds)
        actuals_arr = np.array(actuals)
        
        dir_acc = np.mean((preds_arr > 0) == (actuals_arr > 0)) * 100
        mae = mean_absolute_error(actuals_arr, preds_arr) * 100
        rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr)) * 100
        
        total_preds.extend(preds)
        total_actuals.extend(actuals)
        
        results.append({
            '종목': stock_name,
            '방향정확도': round(dir_acc, 2),
            '수익률': round(total_return, 2),
            '거래횟수': trades,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        })
        
        print(f"{stock_name:12s} {dir_acc:8.2f}% {total_return:8.2f}% {trades:8d}회 {mae:8.4f} {rmse:8.4f}")
    
    # 전체 통계
    total_dir_acc = np.mean((np.array(total_preds) > 0) == (np.array(total_actuals) > 0)) * 100
    avg_mae = mean_absolute_error(total_actuals, total_preds) * 100
    avg_rmse = np.sqrt(mean_squared_error(total_actuals, total_preds)) * 100
    
    print("="*75)
    print(f"📊 전체 평균 방향 정확도 : {total_dir_acc:.2f}%")
    print(f"🎯 평균 수익률           : {pd.DataFrame(results)['수익률'].mean():.2f}%")
    print(f"📉 평균 MAE              : {avg_mae:.4f}%")
    print(f"📉 평균 RMSE             : {avg_rmse:.4f}%")

if __name__ == "__main__":
    run_improved_backtest()