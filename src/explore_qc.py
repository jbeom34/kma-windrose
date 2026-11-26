"""풍향 오자료 규칙을 정하기 위한 탐색: (1) 같은 방위 연속 길이, (2) 월별 벡터 평균 풍향."""
import sys

import numpy as np
import pandas as pd

from windrose_core import DIR_NAMES, classify, load_hourly

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
df["TM"] = pd.to_datetime(df["TM"])
d = classify(df.sort_values("TM").reset_index(drop=True))


def run_stats(d, mask, label, top=10):
    """mask가 참인 시각 중, 같은 방위가 1시간 간격으로 이어진 구간을 찾아 길이를 요약한다."""
    same = (d["DIR"] == d["DIR"].shift()) & (d["TM"].diff() == pd.Timedelta(hours=1))
    cont = mask & mask.shift(fill_value=False) & same
    rid = (~cont).cumsum()
    runs = (d[mask].groupby(rid[mask])
            .agg(START=("TM", "first"), END=("TM", "last"), HOURS=("TM", "size"),
                 DIR=("DIR", "first"), WS_MEAN=("WS", "mean")))
    q = runs["HOURS"].quantile([0.5, 0.9, 0.99, 0.999])
    print(f"\n=== {label}: 같은 방위가 이어진 구간 {len(runs)}개 ===")
    print("길이(시간) 50/90/99/99.9% 분위수:", [int(x) for x in q], "/ 최대", int(runs["HOURS"].max()))
    t = runs.sort_values("HOURS", ascending=False).head(top).copy()
    t["방위"] = t["DIR"].map(lambda k: DIR_NAMES[int(k)])
    t["WS_MEAN"] = t["WS_MEAN"].round(1)
    print(t[["START", "END", "HOURS", "방위", "WS_MEAN"]].to_string(index=False))


run_stats(d, d["SPD"] > 0, "비정온(풍속 0.5 이상)")
run_stats(d, d["WS"] >= 3.4, "풍속 3.4 m/s 이상")

# (2) 월별 벡터 평균 풍향 (비정온 시각만, 방위 번호 x 22.5도)
m = d[d["SPD"] > 0].copy()
a = np.deg2rad(m["DIR"] * 22.5)
m["S"], m["C"] = np.sin(a), np.cos(a)
mon = m.groupby(m["TM"].dt.to_period("M")).agg(S=("S", "mean"), C=("C", "mean"), N=("S", "size"))
mon["평균풍향"] = (np.degrees(np.arctan2(mon["S"], mon["C"])) % 360).round(0)
mon["집중도R"] = np.hypot(mon["S"], mon["C"]).round(2)

base = mon[mon.index.year <= 2025]               # 같은 달끼리 평균 벡터를 만든다
clim = base.groupby(base.index.month)[["S", "C"]].mean()
clim_dir = np.degrees(np.arctan2(clim["S"], clim["C"])) % 360
mon["기준풍향"] = pd.Series(mon.index.month, index=mon.index).map(clim_dir).round(0)
mon["차이(도)"] = ((mon["평균풍향"] - mon["기준풍향"] + 180) % 360 - 180).round(0)

cols = ["평균풍향", "기준풍향", "차이(도)", "집중도R", "N"]
print("\n=== 같은 달 평균 풍향에서 가장 많이 벗어난 달 10개 ===")
print(mon.reindex(mon["차이(도)"].abs().sort_values(ascending=False).index).head(10)[cols].to_string())
for s, e in [("2018-07", "2019-01"), ("2025-09", "2026-02")]:
    print(f"\n--- 풍속계 교체 전후 {s} ~ {e} ---")
    print(mon.loc[s:e, cols].to_string())
