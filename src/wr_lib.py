"""바람장미 분석 공통 함수 (report.py와 이후 웹 서버에서 함께 쓴다)."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from windrose_core import DB, DIR_NAMES, LOCAL_DB, classify, load_hourly

WR_DB = Path("data/windrose.db")
GNAMES = ["정온"] + DIR_NAMES + ["강풍(8.0↑)"]
MIN_DIFF = np.array([2.0] * 17 + [1.0])   # 원인 칸 최소 차이(%p): 방위·정온 2, 강풍 1
Z_MIN, SD_FLOOR = 3.0, 0.5
N_BASE, MIN_BASE = 10, 6                  # 기준 연도 수, 최소 필요 연도 수
FRACS = np.arange(-3.0, 3.001, 0.125)     # 회전 탐색 범위(방위 칸, 1칸 = 22.5도)
MIN_WS = 2.0                              # 지점 비교에 쓰는 최소 풍속(m/s)


# ---------- 분포 계산 ----------
def jsd(a, b):
    """두 분포의 Jensen-Shannon 거리(0이 같음)."""
    m = (a + b) / 2

    def kl(x, y):
        k = x > 0
        return float((x[k] * np.log(x[k] / y[k])).sum())
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def wrap(x):
    """각도를 -180~180도로 맞춘다."""
    return (x + 180) % 360 - 180


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


# ---------- DB 읽기 ----------
def available_stations():
    """두 DB(이 프로젝트, 기존 프로젝트)에 시간자료가 있는 지점 목록."""
    out = set()
    for db in (LOCAL_DB, DB):
        if Path(db).exists():
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                try:
                    out |= {int(r[0]) for r in conn.execute("SELECT DISTINCT STN FROM asos_hourly_clean")}
                except sqlite3.OperationalError:
                    pass
    return sorted(out)


def has_monthly(stn):
    """월별 집계와 QC 표(rose_qc)가 이미 있는지."""
    if not WR_DB.exists():
        return False
    with sqlite3.connect(WR_DB) as conn:
        try:
            return conn.execute("SELECT 1 FROM rose_qc WHERE STN=? LIMIT 1", (stn,)).fetchone() is not None
        except sqlite3.OperationalError:
            return False


def load_month_matrix(stn):
    """rose_cells -> 월별 65칸 비율 표 p, rose_qc -> (연, 월) 색인 표 lab."""
    with sqlite3.connect(WR_DB) as conn:
        cells = pd.read_sql("SELECT YEAR, MONTH, SPD, DIR, HOURS FROM rose_cells WHERE STN=?",
                            conn, params=(stn,))
        qc = pd.read_sql("SELECT * FROM rose_qc WHERE STN=?", conn, params=(stn,))
    cells = cells[cells["DIR"] != -2].copy()
    cells["CELL"] = np.where(cells["DIR"] == -1, 0, 1 + cells["DIR"] * 4 + (cells["SPD"] - 1))
    vec = (cells.pivot_table(index=["YEAR", "MONTH"], columns="CELL", values="HOURS",
                             aggfunc="sum", fill_value=0).reindex(columns=range(65), fill_value=0))
    return vec.div(vec.sum(axis=1), axis=0), qc.set_index(["YEAR", "MONTH"])


def good_years(p, lab, year, month):
    """직전 N_BASE년 중 '관측 의심'과 '자료 부족'이 아닌 같은 달의 연도."""
    cand = [y for y in range(year - N_BASE, year) if (y, month) in p.index and (y, month) in lab.index]
    return [y for y in cand
            if lab.loc[(y, month), "QC_LABEL"] != "관측 의심" and lab.loc[(y, month), "DATA_LOW"] == 0]


# ---------- 판정 ----------
def evaluate(t, B):
    """올해 분포 t를 기준 연도 분포들 B와 비교: 순위, 거리비, 판정, 원인 칸."""
    d_t = jsd(t, B.mean(axis=0))
    loo = [jsd(B[i], np.delete(B, i, axis=0).mean(axis=0)) for i in range(len(B))]
    rank = 1 + sum(x > d_t for x in loo)
    gt, gB = groups(t[None, :])[0], groups(B)
    mu, sd = gB.mean(axis=0), np.maximum(gB.std(axis=0, ddof=1), SD_FLOOR)
    z, diff = (gt - mu) / sd, gt - mu
    hit = np.where((np.abs(z) >= Z_MIN) & (np.abs(diff) >= MIN_DIFF))[0]
    hit = hit[np.argsort(-np.abs(z[hit]))][:3]
    causes = ", ".join(f"{GNAMES[i]} {diff[i]:+.1f}%p(z{z[i]:+.1f})" for i in hit)
    verdict = "특이" if rank == 1 and len(hit) else ("주목" if rank <= 2 else "범위 내")
    return {"rank": int(rank), "n": len(B) + 1, "ratio": float(d_t / max(loo)),
            "verdict": verdict, "causes": causes or "-"}


def month_rotation(p, lab, year, month):
    """그 달의 16방위 분포를 기준 평균에 가장 가깝게 만드는 회전량(칸). 기준이 부족하면 None."""
    good = good_years(p, lab, year, month)
    if len(good) < MIN_BASE or (year, month) not in p.index:
        return None
    base = marg16(np.array([p.loc[(y, month)].values for y in good]).mean(axis=0))
    obs = marg16(p.loc[(year, month)].values)
    return float(FRACS[int(np.argmin([jsd(base, rotate16(obs, f)) for f in FRACS]))])


# ---------- 지점 비교 ----------
def angles(stn):
    """비정온·풍속 MIN_WS 이상 시각의 풍향 각도(도) 시리즈."""
    df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
    if df.empty:
        return pd.Series(dtype=float)
    df["TM"] = pd.to_datetime(df["TM"])
    d = classify(df.dropna(subset=["WS"]))
    d = d[(d["SPD"] > 0) & (d["WS"] >= MIN_WS) & d["DIR"].notna()]
    return d.set_index("TM")["DIR"] * 22.5


def weekly_yoy(target, refs, start, end, min_n=30):
    """대상과 기준 지점들의 같은 시각 풍향 차이를 주 단위로 내고, 52주 전 같은 주와의 변화를 구한다."""
    tg = angles(target)
    parts = []
    for r in refs:
        ra = angles(r)
        if ra.empty:
            continue
        m = pd.concat([tg.rename("A"), ra.rename("B")], axis=1, join="inner")
        diff = np.deg2rad(m["A"] - m["B"])
        parts.append(pd.DataFrame({"S": np.sin(diff), "C": np.cos(diff)}, index=m.index))
    if not parts:
        return pd.DataFrame()
    t = pd.concat(parts).sort_index()

    def weekly(frame):
        sel = frame[(frame.index >= start) & (frame.index < end)]
        wk = (sel.index - start) // pd.Timedelta(days=7)
        g = sel.groupby(wk)
        deg = g.apply(lambda x: float(np.degrees(np.arctan2(x["S"].mean(), x["C"].mean()))))
        return pd.DataFrame({"deg": deg, "N": g.size()})

    cur = weekly(t)
    prev = weekly(t.set_index(t.index + pd.Timedelta(days=364)))
    out = cur.join(prev, rsuffix="_prev", how="inner")
    out = out[(out["N"] >= min_n) & (out["N_prev"] >= min_n)].copy()
    out["week_start"] = [start + pd.Timedelta(days=7 * int(w)) for w in out.index]
    out["change"] = wrap(out["deg"] - out["deg_prev"])
    return out.reset_index(drop=True)


def detect_step(wk, min_seg=4, min_deg=20.0, min_t=4.0):
    """주별 변화에서 앞뒤 평균 차이가 가장 뚜렷한 시점을 찾는다(앞뒤 최소 min_seg주)."""
    x = wk["change"].to_numpy(dtype=float)
    n, best = len(x), None
    for k in range(min_seg, n - min_seg + 1):
        a, b = x[:k], x[k:]
        resid = np.concatenate([a - a.mean(), b - b.mean()])
        sd = resid.std(ddof=2)
        t = abs(b.mean() - a.mean()) / (sd * np.sqrt(1 / len(a) + 1 / len(b))) if sd > 0 else 0.0
        if best is None or t > best["t"]:
            best = {"k": k, "date": wk["week_start"].iloc[k], "before": float(a.mean()),
                    "after": float(b.mean()), "delta": float(b.mean() - a.mean()), "t": float(t)}
    if best:
        best["is_step"] = abs(best["delta"]) >= min_deg and best["t"] >= min_t
    return best
