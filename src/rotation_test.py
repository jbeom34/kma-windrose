"""방위 회전 테스트: 연도(또는 달)의 16방위 분포를 몇 도 돌리면 평소 분포와 가장 비슷해지는지 찾는다.
기준에서 크게 벗어난 해가 풍향계 방위 오차(회전) 하나로 설명되는지 확인하는 용도.
사용법: python src/rotation_test.py [지점번호 ...]   (기본: 159 155)
"""
import sys

import numpy as np
import pandas as pd

from windrose_core import DIR_NAMES, classify, load_hourly

stns = [int(x) for x in sys.argv[1:]] or [159, 155]
FRACS = np.arange(-2.0, 2.01, 0.25)        # 회전량(방위 칸 단위, 1칸 = 22.5도)
I0 = int(np.argmin(np.abs(FRACS)))         # 회전 없음의 위치
pd.set_option("display.width", 200)


def jsd(a, b):
    """두 분포의 Jensen-Shannon 거리(0이 같음)."""
    m = (a + b) / 2

    def kl(x, y):
        k = x > 0
        return float((x[k] * np.log(x[k] / y[k])).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def rotate(p, f):
    """분포 p(길이 16)를 f칸만큼 시계 방향(+)으로 돌린다. 소수는 이웃 칸에 나눠 담는다."""
    k = int(np.floor(f))
    r = f - k
    a = np.roll(p, k)
    return (1 - r) * a + r * np.roll(a, 1)


def dist16(g):
    """비정온 시각의 16방위 비율(합 1)."""
    mv = g[g["SPD"] > 0].dropna(subset=["DIR"])
    c = np.bincount(mv["DIR"].astype(int), minlength=16).astype(float)
    return c / c.sum()


def best(base, obs):
    """obs를 돌려서 base와 가장 비슷해지는 회전을 찾는다.
    반환: (추정 회전 도: + = 관측이 시계 방향으로 돌아가 있음, 보정 전 JSD, 보정 후 JSD)"""
    sc = [jsd(base, rotate(obs, f)) for f in FRACS]
    i = int(np.argmin(sc))
    return -FRACS[i] * 22.5, sc[I0], sc[i]


for stn in stns:
    df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d["Y"], d["M"] = d["TM"].dt.year, d["TM"].dt.month
    yearly = {y: dist16(g) for y, g in d.groupby("Y")}

    Y = np.array([v for y, v in sorted(yearly.items()) if y <= 2025]) * 100
    t = pd.DataFrame({"기준평균": Y.mean(axis=0), "기준최소": Y.min(axis=0), "기준최대": Y.max(axis=0),
                      "2025": yearly[2025] * 100, "2026": yearly[2026] * 100}, index=DIR_NAMES)
    t["2026-평균"] = t["2026"] - t["기준평균"]
    print(f"\n=== 지점 {stn}: 방위별 비율(%, 비정온 시간 기준, 기준=2015~2025) ===")
    print(t.round(1).to_string())

    rows = []
    for y in sorted(yearly):
        base = np.mean([v for o, v in yearly.items() if o != y and o <= 2025], axis=0)
        deg, j0, j1 = best(base, yearly[y])
        rows.append({"연도": y, "JSD 보정전": round(j0, 4), "추정 회전(도)": round(deg, 1),
                     "JSD 보정후": round(j1, 4), "감소%": round((1 - j1 / j0) * 100)})
    print(f"\n=== 지점 {stn}: 연도별 회전 테스트 (+ = 관측이 시계 방향으로 돌아가 있음) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    mon = {k: dist16(g) for k, g in d.groupby(["Y", "M"])}
    rows = []
    for m in range(1, 13):
        if (2026, m) not in mon:
            continue
        base = np.mean([mon[(y, m)] for y in range(2015, 2026) if (y, m) in mon], axis=0)
        deg, j0, j1 = best(base, mon[(2026, m)])
        rows.append({"월": m, "JSD 보정전": round(j0, 4), "추정 회전(도)": round(deg, 1),
                     "JSD 보정후": round(j1, 4), "감소%": round((1 - j1 / j0) * 100)})
    print(f"\n=== 지점 {stn}: 2026년 월별 회전 테스트 ===")
    print(pd.DataFrame(rows).to_string(index=False))
