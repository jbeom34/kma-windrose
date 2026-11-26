"""한 지점(T)의 풍향이 기준 지점들(R)에 대해 언제 계단처럼 변했는지 주 단위로 본다.
사용법: python src/step_timeline.py 159 --ref 255 152 155 --from 2025-11-03 --to 2026-03-16 --mark 2025-12-26 2026-01-30
각 주의 값 = T와 기준 지점들의 같은 시각 풍향 차이(도)를 합쳐 낸 순환 평균(두 지점 모두 풍속 2 m/s 이상인 시각만).
같은 주의 52주 전 값을 옆에 두어 계절 효과를 걸러낸다. 변화 = 올해 - 작년(도).
"""
import argparse

import numpy as np
import pandas as pd

from windrose_core import classify, load_hourly

ap = argparse.ArgumentParser()
ap.add_argument("target", type=int)
ap.add_argument("--ref", nargs="+", type=int, required=True)
ap.add_argument("--from", dest="start", required=True, help="시작일(월요일 권장)")
ap.add_argument("--to", dest="end", required=True)
ap.add_argument("--mark", nargs="*", default=[], help="표시할 날짜(작업일 등)")
args = ap.parse_args()
MIN_WS = 2.0
pd.set_option("display.width", 200)


def angles(stn):
    df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d = d[(d["SPD"] > 0) & (d["WS"] >= MIN_WS) & d["DIR"].notna()]
    return d.set_index("TM")["DIR"] * 22.5


def wrap(x):
    return (x + 180) % 360 - 180


tg = angles(args.target)
parts = []
for r in args.ref:
    m = pd.concat([tg.rename("A"), angles(r).rename("B")], axis=1, join="inner")
    diff = np.deg2rad(m["A"] - m["B"])
    parts.append(pd.DataFrame({"S": np.sin(diff), "C": np.cos(diff)}, index=m.index))
t = pd.concat(parts).sort_index()

start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
shift = pd.Timedelta(days=364)               # 52주 전


def weekly(frame, lo, hi):
    sel = frame[(frame.index >= lo) & (frame.index < hi)]
    wk = (sel.index - lo) // pd.Timedelta(days=7)
    g = sel.groupby(wk)
    deg = g.apply(lambda x: float(np.degrees(np.arctan2(x["S"].mean(), x["C"].mean()))))
    return pd.DataFrame({"deg": deg, "N": g.size()})


cur = weekly(t, start, end)
prev = weekly(t.set_index(t.index + shift), start, end)
marks = [pd.Timestamp(x) for x in args.mark]
rows = []
for w in sorted(cur.index):
    ws = start + pd.Timedelta(days=7 * int(w))
    we = ws + pd.Timedelta(days=7)
    c = cur.loc[w]
    p = prev.loc[w] if w in prev.index else None
    rows.append({
        "주 시작": f"{ws:%Y-%m-%d}",
        "올해(도)": round(c["deg"]), "N": int(c["N"]),
        "작년(도)": round(p["deg"]) if p is not None else None,
        "N작년": int(p["N"]) if p is not None else None,
        "변화(올해-작년)": round(wrap(c["deg"] - p["deg"])) if p is not None else None,
        "표시": " ".join(f"◀{mk:%m-%d}" for mk in marks if ws <= mk < we),
    })
print(f"지점 {args.target} 대 기준 {args.ref}: 주별 풍향 차이(T - R)")
print(pd.DataFrame(rows).to_string(index=False))
