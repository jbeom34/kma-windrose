"""지점 사이의 같은 시각 풍향 차이(순환 평균)로 방위 편향이 생긴 시점과 지점을 찾는다.
사용법: python src/pair_offset.py [지점번호 ...] [--from YYYY-MM] [--to YYYY-MM]
예:     python src/pair_offset.py 159 155 255 --from 2023-10 --to 2024-08
차이 = 앞 지점 - 뒤 지점(도). 지역 날씨는 두 지점에 같은 영향을 주므로,
한쪽 센서의 방위가 바뀌면 그 쌍의 차이가 계단처럼 변한다.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from windrose_core import classify, load_hourly

ap = argparse.ArgumentParser()
ap.add_argument("stns", nargs="*", type=int, default=[159, 155, 255])
ap.add_argument("--from", dest="start", default="2025-07", help="월별 표의 시작 월")
ap.add_argument("--to", dest="end", default="2100-12", help="월별 표의 끝 월")
args = ap.parse_args()

MIN_WS = 2.0                      # 두 지점 모두 풍속이 이 값 이상인 시각만 쓴다(약풍은 풍향 잡음이 큼)
pd.set_option("display.width", 220)

data = {}
for s in args.stns:
    df = load_hourly(s, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    if df.empty:
        print(f"지점 {s}: 자료가 없어 건너뜁니다.")
        continue
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d = d[(d["SPD"] > 0) & (d["WS"] >= MIN_WS) & d["DIR"].notna()]
    data[s] = d.set_index("TM")["DIR"] * 22.5     # 방위 번호 -> 각도


def summarize(g):
    """풍향 차이의 순환 평균(도)과 시간 수."""
    s, c = g["S"].mean(), g["C"].mean()
    return pd.Series({"deg": np.degrees(np.arctan2(s, c)), "N": len(g)})


lo, hi = pd.Period(args.start, "M"), pd.Period(args.end, "M")
ytab, mtab = {}, {}
for a, b in itertools.combinations(data, 2):
    m = pd.concat([data[a].rename("A"), data[b].rename("B")], axis=1, join="inner")
    diff = np.deg2rad(m["A"] - m["B"])
    tmp = pd.DataFrame({"S": np.sin(diff), "C": np.cos(diff)})
    name = f"{a}-{b}"
    y = tmp.groupby(tmp.index.year).apply(summarize)
    mo = tmp.groupby(tmp.index.to_period("M")).apply(summarize)
    mo = mo[(mo.index >= lo) & (mo.index <= hi)]
    ytab[name] = y.apply(lambda r: f"{r['deg']:+.0f} ({int(r['N'])})", axis=1)
    mtab[name] = mo.apply(lambda r: f"{r['deg']:+.0f} ({int(r['N'])})", axis=1)

print("=== 연도별 같은 시각 풍향 차이(도) (사용 시간 수) ===")
print(pd.DataFrame(ytab).to_string())
print(f"\n=== {args.start} ~ {args.end} 월별 ===")
print(pd.DataFrame(mtab).to_string())
