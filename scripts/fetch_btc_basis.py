"""
fetch_btc_basis.py
------------------
매일 GitHub Actions에서 실행.
- 현물: Yahoo Finance BTC-USD (yfinance)
- 선물: Nasdaq Data Link REST API 직접 호출 (CHRIS/CME_BTC1)
  → nasdaqdatalink 라이브러리 미사용 (헤더 오염 이슈 회피)
- Basis 5일 스무딩 + 730일 롤링 Z-Score
- data/btc_basis.json 저장
"""

import os, json, requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from io import StringIO

# ── 설정 ─────────────────────────────────────────────────────────────────────
NASDAQ_KEY   = os.environ.get("NASDAQ_API_KEY", "").strip().replace("\n","").replace("\r","")
START_DATE   = "2017-12-01"
OUTPUT_PATH  = "data/btc_basis.json"
ROLL_WINDOW  = 730    # Z-Score 롤링 윈도우 (2년)
SMOOTH_DAYS  = 5      # Basis 이동평균 스무딩
SPIKE_THRESH = 10.0   # 스파이크 필터 (±10%)

print(f"🔑 API 키 확인: 길이={len(NASDAQ_KEY)}, 끝문자={repr(NASDAQ_KEY[-3:]) if NASDAQ_KEY else 'EMPTY'}")

# ── CME 만기일 마스킹 (매월 마지막 금요일 ±2일) ───────────────────────────────
def get_mask_dates(start, end):
    mask = set()
    for d in pd.date_range(start, end, freq='MS'):
        last_day = d + pd.offsets.MonthEnd(0)
        offset = (last_day.weekday() - 4) % 7
        last_friday = last_day - pd.Timedelta(days=offset)
        for delta in range(-2, 3):
            mask.add((last_friday + pd.Timedelta(days=delta)).date())
    return mask

# ── 1. 현물 (Yahoo Finance) ───────────────────────────────────────────────────
print("📥 [1/3] 현물 (Yahoo Finance BTC-USD)...")
spot_raw = yf.download("BTC-USD", start=START_DATE, progress=False, auto_adjust=True)
if isinstance(spot_raw.columns, pd.MultiIndex):
    spot_raw.columns = spot_raw.columns.get_level_values(0)
spot = spot_raw['Close'].dropna()
spot.index = pd.to_datetime(spot.index).tz_localize(None)
print(f"   {spot.index[0].date()} ~ {spot.index[-1].date()} ({len(spot)}일)")

# ── 2. CME 선물 (Nasdaq Data Link REST API 직접 호출) ────────────────────────
print("📥 [2/3] CME 선물 (Nasdaq Data Link REST API)...")
url = (
    f"https://data.nasdaq.com/api/v3/datasets/CHRIS/CME_BTC1.csv"
    f"?start_date={START_DATE}"
    f"&api_key={NASDAQ_KEY}"
)
resp = requests.get(url, timeout=30)
print(f"   HTTP 상태: {resp.status_code}")

if resp.status_code != 200:
    raise RuntimeError(f"Nasdaq API 실패: {resp.status_code} {resp.text[:300]}")

df = pd.read_csv(StringIO(resp.text), parse_dates=['Date'], index_col='Date')
df.index = pd.to_datetime(df.index).tz_localize(None)
df = df.sort_index()
print(f"   컬럼: {list(df.columns)}")

# Settle 컬럼 우선 사용
col = 'Settle' if 'Settle' in df.columns else df.columns[-1]
btc1 = df[col].dropna()
print(f"   사용 컬럼: {col}")
print(f"   {btc1.index[0].date()} ~ {btc1.index[-1].date()} ({len(btc1)}일)")

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
    "settings": {
        "roll_window":  ROLL_WINDOW,
        "smooth_days":  SMOOTH_DAYS,
        "spike_thresh": SPIKE_THRESH
    },
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
print(f"   Basis:   {len(vb)}일 유효 | {min(vb):.3f}% ~ {max(vb):.3f}%")
print(f"   Z-Score: {len(vz)}일 유효 | {min(vz):.2f} ~ {max(vz):.2f}")
print(f"   🔴 과열(Z≥+2): {sum(1 for z in vz if z >= 2)}일")
print(f"   🟢 공포(Z≤-2): {sum(1 for z in vz if z <= -2)}일")
