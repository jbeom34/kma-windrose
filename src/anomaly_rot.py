"""회전 보정 후 특이점 탐지: 센서 작업으로 방위가 바뀐 달을 보정한 뒤 직전 10년 같은 달과 비교한다.
사용법: python src/anomaly_rot.py [지점번호] [연도] --after YYYY-MM
예:     python src/anomaly_rot.py 159 2026 --after 2026-02
--after 이후의 달은 '월별 최적 회전량의 중앙값'만큼 방위를 되돌린 분포로 다시 비교한다.
먼저 build_monthly.py와 qc_wind.py를 실행해 둘 것.
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

from windrose_core import DIR_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("stn", type=int)
ap.add_argument("year", type=int)
ap.add_argument("--after", required=True, help="방위가 바뀐 뒤 첫 달(YYYY-MM)")
args = ap.parse_args()
stn, year = args.stn, args.year
after = tuple(int(x) for x in args.after.split("-"))

N_BASE, MIN_BASE = 10, 6
Z_MIN, SD_FLOOR = 3.0, 0.5
GNAMES = ["정온"] + DIR_NAMES + ["강풍(8.0↑)"]
MIN_DIFF = np.array([2.0] * 17 + [1.0])
FRACS = np.arange(-3.0, 3.001, 0.125)      # 회전량(방위 칸 단위, 1칸 = 22.5도)

with sqlite3.connect("data/windrose.db") as conn:
    cells = pd.read_sql("SELECT YEAR, MONTH, SPD, DIR, HOURS FROM rose_cells WHERE STN=?",
                        conn, params=(stn,))
    qc = pd.read_sql("SELECT YEAR, MONTH, QC_LABEL, DATA_LOW FROM rose_qc WHERE STN=?",
                     conn, params=(stn,))
lab = qc.set_index(["YEAR", "MONTH"])

cells = cells[cells["DIR"] != -2].copy()
cells["CELL"] = np.where(cells["DIR"] == -1, 0, 1 + cells["DIR"] * 4 + (cells["SPD"] - 1))
vec = (cells.pivot_table(index=["YEAR", "MONTH"], columns="CELL", values="HOURS",
                         aggfunc="sum", fill_value=0).reindex(columns=range(65), fill_value=0))
p = vec.div(vec.sum(axis=1), axis=0)


def jsd(a, b):
    """두 분포의 Jensen-Shannon 거리(0이 같음)."""
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


def rotate16(v, f):
    """길이 16 벡터를 f칸(+ = 시계 방향) 돌린다. 소수는 이웃 칸에 나눠 담는다."""
    k = int(np.floor(f))
    r = f - k
    a = np.roll(v, k)
    return (1 - r) * a + r * np.roll(a, 1)


def rotate_cells(t, f):
    """65칸 분포의 방위 축을 풍속 계급별로 f칸 돌린다(정온 칸은 그대로)."""
    out = t.copy()
    for s in range(4):
        idx = 1 + np.arange(16) * 4 + s
        out[idx] = rotate16(t[idx], f)
    return out


def marg16(t):
    """65칸 분포 -> 비정온 16방위 비율(합 1)."""
    d = np.array([t[1 + k * 4: 5 + k * 4].sum() for k in range(16)])
    return d / d.sum()


def good_years(month):
    cand = [y for y in range(year - N_BASE, year) if (y, month) in p.index]
    return [y for y in cand
            if lab.loc[(y, month), "QC_LABEL"] != "관측 의심" and lab.loc[(y, month), "DATA_LOW"] == 0]


def evaluate(t, good, B):
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
    return rank, d_t / max(loo), verdict, causes or "-"


months = [m for m in range(1, 13) if (year, m) in p.index]
post = [m for m in months if (year, m) >= after]

# 1) 작업 이후 달마다 최적 회전량을 구하고 중앙값을 공통 보정량으로 쓴다
est = {}
for m in post:
    good = good_years(m)
    if len(good) < MIN_BASE:
        continue
    B = np.array([p.loc[(y, m)].values for y in good]).mean(axis=0)
    obs16, base16 = marg16(p.loc[(year, m)].values), marg16(B)
    est[m] = FRACS[int(np.argmin([jsd(base16, rotate16(obs16, f)) for f in FRACS]))]
f_corr = float(np.median(list(est.values()))) if est else 0.0
print(f"지점 {stn}, {year}년, {args.after} 이후 방위 보정")
print("월별 추정 회전(도, + = 관측이 시계 방향으로 돌아가 있음):",
      {m: round(-f * 22.5, 1) for m, f in est.items()})
print(f"공통 보정량(중앙값): {-f_corr * 22.5:+.1f}도\n")

# 2) 원자료와 보정 후를 나란히 비교한다
rows = []
for m in months:
    good = good_years(m)
    if len(good) < MIN_BASE:
        continue
    B = np.array([p.loc[(year, m)].values * 0 + p.loc[(y, m)].values for y in good])
    t = p.loc[(year, m)].values
    r0 = evaluate(t, good, B)
    fixed = (year, m) >= after
    r1 = evaluate(rotate_cells(t, f_corr), good, B) if fixed else r0
    note = [f"QC {lab.loc[(year, m), 'QC_LABEL']}"] if lab.loc[(year, m), "QC_LABEL"] != "정상" else []
    if fixed:
        note.append("보정 적용")
    rows.append({"월": m, "순위(원)": f"{r0[0]}/{len(good) + 1}", "거리비(원)": round(r0[1], 2), "판정(원)": r0[2],
                 "순위(보정)": f"{r1[0]}/{len(good) + 1}", "거리비(보정)": round(r1[1], 2), "판정(보정)": r1[2],
                 "원인 칸(보정 후)": r1[3], "비고": ", ".join(note) or "-"})
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 80)
print(pd.DataFrame(rows).to_string(index=False))
