"""3단계: 선택 연도의 월별 바람장미를 직전 10년 같은 달과 비교해 특이점 후보를 찾는다.
사용법: python src/anomaly.py [지점번호] [연도]
먼저 build_monthly.py와 qc_wind.py를 실행해 둘 것.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

from windrose_core import DIR_NAMES

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
N_BASE, MIN_BASE = 10, 6                 # 기준 연도 수, 최소 필요 개수
Z_MIN, SD_FLOOR = 3.0, 0.5               # 원인 칸 기준: z 3 이상, 표준편차 하한 0.5%p
GNAMES = ["정온"] + DIR_NAMES + ["강풍(8.0↑)"]
MIN_DIFF = np.array([2.0] * 17 + [1.0])  # 원인 칸의 최소 차이(%p)

with sqlite3.connect("data/windrose.db") as conn:
    cells = pd.read_sql("SELECT YEAR, MONTH, SPD, DIR, HOURS FROM rose_cells WHERE STN=?",
                        conn, params=(stn,))
    qc = pd.read_sql("SELECT YEAR, MONTH, QC_LABEL, DATA_LOW, EXPECTED_HOURS FROM rose_qc WHERE STN=?",
                     conn, params=(stn,))
lab = qc.set_index(["YEAR", "MONTH"])

# 칸 번호: 0=정온, 1~64 = 방위(0~15) x 풍속 계급(1~4)
cells = cells[cells["DIR"] != -2].copy()
cells["CELL"] = np.where(cells["DIR"] == -1, 0, 1 + cells["DIR"] * 4 + (cells["SPD"] - 1))
vec = (cells.pivot_table(index=["YEAR", "MONTH"], columns="CELL", values="HOURS",
                         aggfunc="sum", fill_value=0).reindex(columns=range(65), fill_value=0))
p = vec.div(vec.sum(axis=1), axis=0)     # 월별 비율(합 1)


def jsd(a, b):
    """두 분포의 Jensen-Shannon 거리(자연로그 기준, 0이 같음)."""
    m = (a + b) / 2

    def kl(x, y):
        k = x > 0
        return float((x[k] * np.log(x[k] / y[k])).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def groups(P):
    """65칸 비율 -> 정온, 16방위(풍속 계급 합), 강풍(8.0 이상) 비율(%) 18개."""
    g = np.zeros((P.shape[0], 18))
    g[:, 0] = P[:, 0]
    for k in range(16):
        g[:, 1 + k] = P[:, 1 + 4 * k: 5 + 4 * k].sum(axis=1)
    idx = np.arange(1, 65)
    g[:, 17] = P[:, idx[((idx - 1) % 4) >= 2]].sum(axis=1)
    return g * 100


rows = []
for month in range(1, 13):
    if (year, month) not in p.index:
        continue
    cand = [y for y in range(year - N_BASE, year) if (y, month) in p.index]
    good = [y for y in cand
            if lab.loc[(y, month), "QC_LABEL"] != "관측 의심" and lab.loc[(y, month), "DATA_LOW"] == 0]
    if len(good) < MIN_BASE:
        print(f"{month}월: 기준 연도가 {len(good)}개뿐이라 건너뜀")
        continue
    t = p.loc[(year, month)].values
    B = np.array([p.loc[(y, month)].values for y in good])
    d_t = jsd(t, B.mean(axis=0))
    loo = [jsd(B[i], np.delete(B, i, axis=0).mean(axis=0)) for i in range(len(good))]
    rank = 1 + sum(x > d_t for x in loo)

    gt, gB = groups(t[None, :])[0], groups(B)
    mu, sd = gB.mean(axis=0), np.maximum(gB.std(axis=0, ddof=1), SD_FLOOR)
    z, diff = (gt - mu) / sd, gt - mu
    hit = np.where((np.abs(z) >= Z_MIN) & (np.abs(diff) >= MIN_DIFF))[0]
    hit = hit[np.argsort(-np.abs(z[hit]))][:3]
    causes = ", ".join(f"{GNAMES[i]} {diff[i]:+.1f}%p(z{z[i]:+.1f})" for i in hit)

    verdict = "특이" if rank == 1 and len(hit) else ("주목" if rank <= 2 else "범위 내")
    note = []
    r = lab.loc[(year, month)]
    if r["QC_LABEL"] != "정상":
        note.append(f"QC {r['QC_LABEL']}")
    if r["DATA_LOW"] == 1:
        note.append("자료 부족")
    if r["EXPECTED_HOURS"] < pd.Period(f"{year}-{month:02d}").days_in_month * 24 - 1e-9:
        note.append("미완 월")
    rows.append({"월": month, "기준연도수": len(good), "순위": f"{rank}/{len(good) + 1}",
                 "거리비(올해/기준최대)": round(d_t / max(loo), 2), "판정": verdict,
                 "원인 칸": causes or "-", "비고": ", ".join(note) or "-"})

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(f"지점 {stn}, {year}년 vs 직전 {N_BASE}년 (순위 1 = 올해가 평소에서 가장 멀다)")
print(out.to_string(index=False))
