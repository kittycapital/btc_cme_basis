"""
fetch_btc_basis.py
-----------------
매일 GitHub Actions에서 실행.
Yahoo Finance에서 BTC 현물(BTC-USD)과 CME 선물(BTC=F)을 받아
Basis % 계산 후 data/btc_basis.json 저장.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ── 설정 ────────────────────────────────────────────────────────────────────
START_DATE = "2017-12-01"       # CME BTC 선물 상장일
OUTPUT_PATH = "data/btc_basis.json"
SPIKE_THRESHOLD = 15.0          # 롤오버 왜곡 필터 (±15% 초과 제거)
INTERP_LIMIT = 3                # NaN 보간 최대 일수

# ── 데이터 다운로드 ─────────────────────────────────────────────────────────
print("📥 Yahoo Finance 데이터 다운로드 중...")

spot_raw = yf.download("BTC-USD", start=START_DATE, progress=False, auto_adjust=True)
fut_raw  = yf.download("BTC=F",   start=START_DATE, progress=False, auto_adjust=True)

# MultiIndex 컬럼 처리
def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

spot_raw = flatten(spot_raw)
fut_raw  = flatten(fut_raw)

spot = spot_raw[['Open','High','Low','Close']].dropna(subset=['Close'])
fut  = fut_raw['Close'].dropna()

# ── 날짜 정렬 & Basis 계산 ───────────────────────────────────────────────────
common = spot.index.intersection(fut.index)
spot_c = spot.loc[common]
fut_c  = fut.loc[common]

basis_raw = (fut_c - spot_c['Close']) / spot_c['Close'] * 100

# 롤오버 스파이크 제거
basis_clean = basis_raw.copy()
basis_clean[basis_clean.abs() > SPIKE_THRESHOLD] = np.nan

# 단기 보간 (최대 3일 NaN 메우기)
basis_clean = basis_clean.interpolate(method='time', limit=INTERP_LIMIT)

# ── spot 전체 날짜 기준으로 merge ────────────────────────────────────────────
all_dates = spot.index

def safe(series, idx):
    return float(series[idx]) if idx in series.index and not pd.isna(series[idx]) else None

result = {
    "updated": datetime.utcnow().strftime("%Y-%m-%d"),
    "source": "Yahoo Finance (BTC-USD / BTC=F)",
    "dates":   [d.strftime("%Y-%m-%d") for d in all_dates],
    "opens":   [round(float(spot.loc[d,'Open']),  2) if d in spot.index else None for d in all_dates],
    "highs":   [round(float(spot.loc[d,'High']),  2) if d in spot.index else None for d in all_dates],
    "lows":    [round(float(spot.loc[d,'Low']),   2) if d in spot.index else None for d in all_dates],
    "closes":  [round(float(spot.loc[d,'Close']), 2) if d in spot.index else None for d in all_dates],
    "futures": [round(float(fut_c[d]),  2) if d in fut_c.index  and not pd.isna(fut_c[d])  else None for d in all_dates],
    "basis":   [round(float(basis_clean[d]), 4) if d in basis_clean.index and not pd.isna(basis_clean[d]) else None for d in all_dates],
}

# ── 저장 ────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

# ── 요약 ────────────────────────────────────────────────────────────────────
valid_basis = [b for b in result['basis'] if b is not None]
pos = sum(1 for b in valid_basis if b >= 0)
neg = len(valid_basis) - pos

print(f"✅ 저장 완료: {OUTPUT_PATH}")
print(f"   기간: {result['dates'][0]} ~ {result['dates'][-1]}")
print(f"   스팟 데이터: {len(result['dates'])}일")
print(f"   Basis 유효: {len(valid_basis)}일  (Contango {pos}일 / Backwardation {neg}일)")
print(f"   Basis 범위: {min(valid_basis):.3f}% ~ {max(valid_basis):.3f}%")
