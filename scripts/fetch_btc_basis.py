"""
fetch_btc_basis.py
------------------
매일 GitHub Actions에서 실행.
- 현물: Yahoo Finance BTC-USD
- 선물: Nasdaq Data Link CHRIS/CME_BTC1 (최근월)
- Basis = (선물 - 현물) / 현물 × 100
- 5일 이동평균 스무딩 후 730일 롤링 Z-Score 계산
- data/btc_basis.json 저장
"""

import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import nasdaqdatalink
from datetime import datetime

# ── 설정 ─────────────────────────────────────────────────────────────────────
NASDAQ_KEY   = os.environ.get("NASDAQ_API_KEY", "").strip()
START_DATE   = "2017-12-01"
OUTPUT_PATH  = "data/btc_basis.json"
ROLL_WINDOW  = 730    # Z-Score 롤링 윈도우 (2년)
SMOOTH_DAYS  = 5      # Basis 이동평균 스무딩
SPIKE_THRESH = 10.0   # 롤오버 스파이크 필터 (±10%)

# ── CME 만기일 생성 (매월 마지막 금요일 ±2일 마스킹) ─────────────────────────
def get_mask_dates(start, end):
    mask = set()
    dates = pd.date_range(start, end, freq='MS')  # 매월 1일
    for d in dates:
        # 해당 월의 마지막 금요일
        last_day = d + pd.offsets.MonthEnd(0)
        offset = (last_day.weekday() - 4) % 7  # 금요일까지 거슬러 올라가기
        last_friday = last_day - pd.Timedelta(days=offset)
        for delta in range(-2, 3):
            mask.add((last_friday + pd.Timedelta(days=delta)).date())
    return mask

# ── 1. 현물 ──────────────────────────────────────────────────────────────────
print("📥 [1/3] 현물 (Yahoo Finance BTC-USD)...")
spot_raw = yf.download("BTC-USD", start=START_DATE, progress=False, auto_adjust=True)
if isinstance(spot_raw.columns, pd.MultiIndex):
    spot_raw.columns = spot_raw.columns.get_level_values(0)
spot = spot_raw['Close'].dropna()
spot.index = pd.to_datetime(spot.index).tz_localize(None)
print(f"   {spot.index[0].date()} ~ {spot.index[-1].date()} ({len(spot)}일)")

# ── 2. CME 선물 (Nasdaq Data Link) ───────────────────────────────────────────
print("📥 [2/3] CME 선물 (Nasdaq Data Link CHRIS/CME_BTC1)...")
nasdaqdatalink.ApiConfig.api_key = NASDAQ_KEY
try:
    df = nasdaqdatalink.get("CHRIS/CME_BTC1", start_date=START_DATE)
    col = 'Settle' if 'Settle' in df.columns else df.columns[-1]
    btc1 = df[col].dropna()
    btc1.index = pd.to_datetime(btc1.index).tz_localize(None)
    print(f"   {btc1.index[0].date()} ~ {btc1.index[-1].date()} ({len(btc1)}일), 컬럼: {col}")
except Exception as e:
    raise RuntimeError(f"CME_BTC1 로드 실패: {e}")

# ── 3. Basis 계산 ─────────────────────────────────────────────────────────────
print("🔧 [3/3] Basis & Z-Score 계산...")
common = spot.index.intersection(btc1.index)
spot_c = spot.loc[common]
fut_c  = btc1.loc[common]

basis_raw = (fut_c - spot_c) / spot_c * 100

# 스파이크 제거
basis_raw[basis_raw.abs() > SPIKE_THRESH] = np.nan

# 만기일 ±2일 마스킹
mask = get_mask_dates(START_DATE, spot.index[-1].strftime("%Y-%m-%d"))
for d in basis_raw.index:
    if d.date() in mask:
        basis_raw[d] = np.nan

# 5일 스무딩
basis_smooth = basis_raw.rolling(window=SMOOTH_DAYS, min_periods=3, center=True).mean()

# 730일 Z-Score
roll_mean = basis_smooth.rolling(window=ROLL_WINDOW, min_periods=90).mean()
roll_std  = basis_smooth.rolling(window=ROLL_WINDOW, min_periods=90).std()
zscore    = ((basis_smooth - roll_mean) / roll_std).round(3)

# ── 4. JSON 생성 & 저장 ───────────────────────────────────────────────────────
def to_val(series, idx):
    try:
        val = series.get(idx)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 4)
    except:
        return None

all_dates = spot.index
result = {
    "updated":  datetime.utcnow().strftime("%Y-%m-%d"),
    "source":   "Nasdaq Data Link (CHRIS/CME_BTC1) + Yahoo Finance (BTC-USD)",
    "settings": {"roll_window": ROLL_WINDOW, "smooth_days": SMOOTH_DAYS, "spike_thresh": SPIKE_THRESH},
    "dates":   [d.strftime("%Y-%m-%d") for d in all_dates],
    "closes":  [to_val(spot,         d) for d in all_dates],
    "futures": [to_val(fut_c,        d) for d in all_dates],
    "basis":   [to_val(basis_smooth, d) for d in all_dates],
    "zscore":  [to_val(zscore,       d) for d in all_dates],
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

# ── 요약 ─────────────────────────────────────────────────────────────────────
vb = [b for b in result['basis']  if b is not None]
vz = [z for z in result['zscore'] if z is not None]
print(f"\n✅ 완료: {OUTPUT_PATH}")
print(f"   기간:    {result['dates'][0]} ~ {result['dates'][-1]}")
print(f"   Basis:   {len(vb)}일 | {min(vb):.3f}% ~ {max(vb):.3f}%")
print(f"   Z-Score: {len(vz)}일 | {min(vz):.2f} ~ {max(vz):.2f}")
print(f"   🔴 과열(Z≥+2): {sum(1 for z in vz if z >= 2)}일")
print(f"   🟢 공포(Z≤-2): {sum(1 for z in vz if z <= -2)}일")
