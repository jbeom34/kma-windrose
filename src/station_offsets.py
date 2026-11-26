"""여러 지점의 같은 시각 풍향 차이로 지점별 방위 편차를 추정한다.
사용법: python src/station_offsets.py [지점번호 ...] [--base 2017-2019] [--from YYYY-MM] [--to YYYY-MM]
쌍마다 기준 기간의 풍향 차이를 0으로 본 '쌍별 편차'를 만들고, 지점별 편차(합계 0)를 최소제곱으로 푼다.
편차 + = 그 지점이 다른 지점들 평균보다 시계 방향으로 돌아가 있음. 잔차RMS가 작을수록 잘 설명됨.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from windrose_core import classify, load_hourly

ap = argparse.ArgumentParser()
ap.add_argument("stns", nargs="*", type=int, default=[159, 155, 255, 152])
ap.add_argument("--base", default="2017-2019", help="기준 기간(시작연도-끝연도)")
ap.add_argument("--from", dest="start", default="2025-07")
ap.add_argument("--to", dest="end", default="2100-12")
args = ap.parse_args()
b0, b1 = (int(x) for x in args.base.split("-"))
MIN_WS = 2.0
pd.set_option("display.width", 200)

data = {}
for s in args.stns:
    df = load_hourly(s, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    if df.empty:
        print(f"지점 {s}: 자료가 없어 건너뜁니다.")
        continue
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d = d[(d["SPD"] > 0) & (d["WS"] >= MIN_WS) & d["DIR"].notna()]
    data[s] = d.set_index("TM")["DIR"] * 22.5
ids = list(data)


def circ(g):
    """풍향 차이(sin, cos)의 순환 평균(도)."""
    return float(np.degrees(np.arctan2(g["S"].mean(), g["C"].mean())))


def wrap(x):
    return (x + 180) % 360 - 180


pairs = {}
for a, b in itertools.combinations(ids, 2):
    m = pd.concat([data[a].rename("A"), data[b].rename("B")], axis=1, join="inner")
    diff = np.deg2rad(m["A"] - m["B"])
    t = pd.DataFrame({"S": np.sin(diff), "C": np.cos(diff)}, index=m.index)
    base = circ(t[(t.index.year >= b0) & (t.index.year <= b1)])
    pairs[(a, b)] = {"base": base,
                     "y": t.groupby(t.index.year).apply(circ),
                     "m": t.groupby(t.index.to_period("M")).apply(circ)}


def solve(dev):
    """쌍별 편차 {(a, b): 도} -> (지점별 편차 배열, 잔차 RMS). 지점 편차의 합은 0으로 둔다."""
    n = len(ids)
    rows, y = [], []
    for (a, b), v in dev.items():
        r = np.zeros(n)
        r[ids.index(a)], r[ids.index(b)] = 1.0, -1.0
        rows.append(r)
        y.append(v)
    rows.append(np.ones(n))
    y.append(0.0)
    A, y = np.array(rows), np.array(y)
    o = np.linalg.lstsq(A, y, rcond=None)[0]
    return o, float(np.sqrt(np.mean((A @ o - y)[:-1] ** 2)))


def table(key, periods):
    out = []
    for p in periods:
        dev = {pr: wrap(v[key][p] - v["base"]) for pr, v in pairs.items() if p in v[key].index}
        if len(dev) < len(pairs):
            continue
        o, rms = solve(dev)
        out.append({"기간": str(p), **{str(s): round(float(x)) for s, x in zip(ids, o)},
                    "잔차RMS": round(rms)})
    return pd.DataFrame(out).to_string(index=False)


last = max(s.index.max() for s in data.values()).to_period("M")
lo, hi = pd.Period(args.start, "M"), min(pd.Period(args.end, "M"), last)
print(f"=== 연도별 지점 방위 편차(도), 기준 {args.base} ===")
print(table("y", range(2015, last.year + 1)))
print(f"\n=== {lo} ~ {hi} 월별 ===")
print(table("m", pd.period_range(lo, hi, freq="M")))
