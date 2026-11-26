"""연도별 프로필: 정온 비율, 평균풍속, 주요 방위 비율의 연도별 추세를 지점별로 출력한다.
사용법: python src/yearly_profile.py [지점번호 ...]   (기본: 159 155)
"""
import sys

import numpy as np
import pandas as pd

from windrose_core import DIR_NAMES, classify, load_hourly

stns = [int(x) for x in sys.argv[1:]] or [159, 155]
SHOW = ["북동", "동북동", "남동", "남남서", "서북서"]
pd.set_option("display.width", 220)

for stn in stns:
    df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d["Y"] = d["TM"].dt.year
    rows = []
    for y, g in d.groupby("Y"):
        n = len(g)
        row = {"연도": y, "시간": n, "평균풍속": round(g["WS"].mean(), 2),
               "정온%": round((g["SPD"] == 0).mean() * 100, 1),
               "8.0이상%": round((g["SPD"] >= 3).mean() * 100, 2)}
        for name in SHOW:
            row[name + "%"] = round((g["DIR"] == DIR_NAMES.index(name)).sum() / n * 100, 1)
        mv = g[g["SPD"] > 0]
        a = np.deg2rad(mv["DIR"] * 22.5)
        s, c = np.sin(a).mean(), np.cos(a).mean()
        row["평균풍향"] = int(np.degrees(np.arctan2(s, c)) % 360)
        row["집중도R"] = round(float(np.hypot(s, c)), 2)
        rows.append(row)
    print(f"\n=== 지점 {stn}: 연도별 프로필 (2026년은 11월까지) ===")
    print(pd.DataFrame(rows).to_string(index=False))
