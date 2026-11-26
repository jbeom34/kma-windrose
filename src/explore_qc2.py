"""2단계 탐색 2: (1) 월별 정온 비율, (2) 기준 집중도가 높은 달만의 풍향 편차."""
import sys

import numpy as np
import pandas as pd

from windrose_core import classify, load_hourly

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
df["TM"] = pd.to_datetime(df["TM"])
d = classify(df.dropna(subset=["WS"]).sort_values("TM").reset_index(drop=True))
d["YM"] = d["TM"].dt.to_period("M")
d["WS_SUSPECT"] = d["WS_SUSPECT"].fillna(0)

# (1) 월별 정온 비율을 같은 달(2015~2025) 중앙값과 비교
calm = d.groupby("YM").agg(VALID=("WS", "size"), CALM=("SPD", lambda s: int((s == 0).sum())),
                           SUSPECT=("WS_SUSPECT", "sum"))
calm["정온%"] = (calm["CALM"] / calm["VALID"] * 100).round(1)
base = calm[calm.index.year <= 2025]
med = base.groupby(base.index.month)["정온%"].median()
calm["기준중앙값%"] = pd.Series(calm.index.month, index=calm.index).map(med)
calm["차이%p"] = (calm["정온%"] - calm["기준중앙값%"]).round(1)
cols = ["VALID", "CALM", "SUSPECT", "정온%", "기준중앙값%", "차이%p"]

print("=== 정온 비율이 같은 달 중앙값보다 가장 높은 달 15개 ===")
print(calm.sort_values("차이%p", ascending=False).head(15)[cols].to_string())
for s, e in [("2018-09", "2018-12"), ("2025-10", "2026-01"), ("2026-09", "2026-11")]:
    print(f"\n--- {s} ~ {e} ---")
    print(calm.loc[s:e, cols].to_string())

# (2) 기준 집중도 R이 0.3 이상인 달만의 벡터 평균 풍향 편차
m = d[d["SPD"] > 0].copy()
a = np.deg2rad(m["DIR"] * 22.5)
m["S"], m["C"] = np.sin(a), np.cos(a)
mon = m.groupby("YM").agg(S=("S", "mean"), C=("C", "mean"))
mon["평균풍향"] = np.degrees(np.arctan2(mon["S"], mon["C"])) % 360
mon["month"] = mon.index.month
b = mon[mon.index.year <= 2025]
clim = b.groupby("month")[["S", "C"]].mean()
clim["기준풍향"] = np.degrees(np.arctan2(clim["S"], clim["C"])) % 360
clim["기준R"] = np.hypot(clim["S"], clim["C"])
print("\n=== 달별 기준 집중도 R (낮으면 평균 풍향의 의미가 약함) ===")
print(clim[["기준풍향", "기준R"]].round({"기준풍향": 0, "기준R": 2}).to_string())

mon["기준풍향"] = mon["month"].map(clim["기준풍향"])
mon["기준R"] = mon["month"].map(clim["기준R"])
mon["차이(도)"] = (mon["평균풍향"] - mon["기준풍향"] + 180) % 360 - 180
ok = mon[mon["기준R"] >= 0.3]
q = ok["차이(도)"].abs().quantile([0.5, 0.9, 0.95])
print(f"\n기준 R 0.3 이상인 달 {len(ok)}개월, |차이| 50/90/95% 분위수:", [int(x) for x in q])
top = ok.reindex(ok["차이(도)"].abs().sort_values(ascending=False).index).head(10)
print(top[["평균풍향", "기준풍향", "차이(도)", "기준R"]].round(
    {"평균풍향": 0, "기준풍향": 0, "차이(도)": 0, "기준R": 2}).to_string())
