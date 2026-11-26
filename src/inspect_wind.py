"""기존 kma.db의 풍향(WD)·풍속(WS) 값 형태를 읽기 전용으로 확인한다."""
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path.home() / "kma-anomaly" / "data" / "kma.db"
with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
    df = pd.read_sql("SELECT TM, STN, WD, WS FROM asos_hourly_clean", conn)

print(f"행 수: {len(df)}, 지점: {sorted(df['STN'].unique().tolist())}")
print(f"기간: {df['TM'].min()} ~ {df['TM'].max()}")

print("\n=== 풍향(WD) ===")
print(f"범위: {df['WD'].min()} ~ {df['WD'].max()}, 고유값 {df['WD'].nunique()}개, 결측 {df['WD'].isna().sum()}행")
print("값별 개수:")
print(df["WD"].value_counts(dropna=False).sort_index().to_string())

print("\n=== 풍속(WS) ===")
print(f"범위: {df['WS'].min()} ~ {df['WS'].max()}, 결측 {df['WS'].isna().sum()}행")
print(f"WS == 0: {(df['WS'] == 0).sum()}행, WS <= 0.4: {(df['WS'] <= 0.4).sum()}행")

print("\n=== 정합성 ===")
print("WS == 0일 때 WD 값:")
print(df[df["WS"] == 0]["WD"].value_counts(dropna=False).head(5).to_string())
print(f"\nWS > 0인데 WD 결측: {((df['WS'] > 0) & df['WD'].isna()).sum()}행")
print(f"WS 결측인데 WD 있음: {(df['WS'].isna() & df['WD'].notna()).sum()}행")
