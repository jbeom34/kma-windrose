"""한 해 안에서 평소와 달라지기 시작한 시점 찾기: 반월(1~15일, 16일~말일) 단위 비교.
사용법: python src/changepoint.py [지점번호] [연도]
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

from windrose_core import classify, load_hourly

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
MIN_HOURS, MIN_BASE = 120, 6

with sqlite3.connect("data/windrose.db") as conn:
    qc = pd.read_sql("SELECT YEAR, MONTH, QC_LABEL, DATA_LOW FROM rose_qc WHERE STN=?",
                     conn, params=(stn,))
bad = {(r.YEAR, r.MONTH) for r in qc.itertuples() if r.QC_LABEL == "관측 의심" or r.DATA_LOW == 1}
qlab = {(r.YEAR, r.MONTH): r.QC_LABEL for r in qc.itertuples()}

df = load_hourly(stn, f"{year - 10}-01-01 00:00:00", f"{year + 1}-01-01 00:00:00")
df["TM"] = pd.to_datetime(df["TM"])
d = classify(df.dropna(subset=["WS"]))
d = d[(d["SPD"] == 0) | d["DIR"].notna()].copy()
d["Y"], d["M"] = d["TM"].dt.year, d["TM"].dt.month
d["H"] = np.where(d["TM"].dt.day <= 15, 1, 2)
d["CAT"] = np.where(d["SPD"] == 0, 0, d["DIR"].fillna(0) + 1).astype(int)   # 0=정온, 1~16=방위

cnt = d.groupby(["Y", "M", "H", "CAT"]).size().unstack(fill_value=0).reindex(columns=range(17), fill_value=0)
n = cnt.sum(axis=1)
p = cnt.div(n, axis=0)


def jsd(a, b):
    """두 분포의 Jensen-Shannon 거리(0이 같음)."""
    m = (a + b) / 2

    def kl(x, y):
        k = x > 0
        return float((x[k] * np.log(x[k] / y[k])).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


rows = []
for m in range(1, 13):
    for h in (1, 2):
        key = (year, m, h)
        if key not in p.index or n.loc[key] < MIN_HOURS:
            continue
        ys = [y for y in range(year - 10, year)
              if (y, m, h) in p.index and (y, m) not in bad and n.loc[(y, m, h)] >= MIN_HOURS]
        if len(ys) < MIN_BASE:
            continue
        B = np.array([p.loc[(y, m, h)].values for y in ys])
        t = p.loc[key].values
        dt = jsd(t, B.mean(axis=0))
        loo = [jsd(B[i], np.delete(B, i, axis=0).mean(axis=0)) for i in range(len(ys))]
        rank = 1 + sum(x > dt for x in loo)
        rows.append({"반월": f"{m:02d}월 {'상' if h == 1 else '하'}", "시간": int(n.loc[key]),
                     "기준연도수": len(ys), "순위": f"{rank}/{len(ys) + 1}",
                     "거리비": round(dt / max(loo), 2), "정온%": round(t[0] * 100, 1),
                     "QC": qlab.get((year, m), "-")})

print(f"지점 {stn}, {year}년 반월별 비교 (순위 1 = 평소에서 가장 멀다)")
print(pd.DataFrame(rows).to_string(index=False))
