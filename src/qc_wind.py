"""월별 관측 품질 표시(rose_qc)를 만든다: 정온 비율, 풍향 고착 후보, 자료 부족.
사용법: python src/qc_wind.py [지점번호]   (먼저 build_monthly.py를 실행해 둘 것)
"""
import sqlite3
import sys

import pandas as pd

from windrose_core import classify, load_hourly

CALM_WATCH, CALM_BAD = 5.0, 10.0   # 같은 달 중앙값보다 이만큼(%p) 높으면 관찰 / 관측 의심
STUCK_HOURS, STUCK_WS = 24, 3.4    # 풍속 3.4 이상에서 같은 방위가 24시간 이상 이어지면 고착 후보
LOW_DATA = 0.95                    # 유효 시간이 기대의 95% 미만이면 자료 부족
BASE_LAST_YEAR = 2025              # 기준 기간: 2015 ~ 이 연도
OUT = "data/windrose.db"

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
df["TM"] = pd.to_datetime(df["TM"])
d = classify(df.dropna(subset=["WS"]).sort_values("TM").reset_index(drop=True))
d["YM"] = d["TM"].dt.to_period("M")
d["WS_SUSPECT"] = d["WS_SUSPECT"].fillna(0)


def find_runs(d, mask):
    """mask가 참인 시각 중 같은 방위가 1시간 간격으로 이어진 구간(시작 시각, 길이)."""
    same = (d["DIR"] == d["DIR"].shift()) & (d["TM"].diff() == pd.Timedelta(hours=1))
    cont = mask & mask.shift(fill_value=False) & same
    rid = (~cont).cumsum()
    return d[mask].groupby(rid[mask]).agg(START=("TM", "first"), HOURS=("TM", "size"))


runs = find_runs(d, d["WS"] >= STUCK_WS)
runs = runs[runs["HOURS"] >= STUCK_HOURS].copy()
runs["YM"] = runs["START"].dt.to_period("M")           # 구간은 시작한 달에 넣는다
stuck = runs.groupby("YM")["HOURS"].agg(STUCK_RUNS="size", STUCK_MAX_HOURS="max")

q = d.groupby("YM").agg(VALID_HOURS=("WS", "size"),
                        CALM_HOURS=("SPD", lambda s: int((s == 0).sum())),
                        SUSPECT_HOURS=("WS_SUSPECT", "sum"))
q["SUSPECT_HOURS"] = q["SUSPECT_HOURS"].astype(int)
q["CALM_PCT"] = (q["CALM_HOURS"] / q["VALID_HOURS"] * 100).round(1)
base = q[q.index.year <= BASE_LAST_YEAR]
med = base.groupby(base.index.month)["CALM_PCT"].median()
q["BASE_CALM_PCT"] = pd.Series(q.index.month, index=q.index).map(med)
q["CALM_DIFF"] = (q["CALM_PCT"] - q["BASE_CALM_PCT"]).round(1)
q = q.join(stuck)
q[["STUCK_RUNS", "STUCK_MAX_HOURS"]] = q[["STUCK_RUNS", "STUCK_MAX_HOURS"]].fillna(0).astype(int)

with sqlite3.connect(OUT) as conn:
    meta = pd.read_sql("SELECT YEAR, MONTH, EXPECTED_HOURS FROM rose_meta WHERE STN=?",
                       conn, params=(stn,))
meta["YM"] = pd.PeriodIndex(meta["YEAR"].astype(str) + "-" + meta["MONTH"].astype(str).str.zfill(2),
                            freq="M")
q = q.join(meta.set_index("YM")["EXPECTED_HOURS"])
q["VALID_PCT"] = (q["VALID_HOURS"] / q["EXPECTED_HOURS"] * 100).round(1)
q["DATA_LOW"] = (q["VALID_PCT"] < LOW_DATA * 100).astype(int)
q["QC_LABEL"] = "정상"
q.loc[q["CALM_DIFF"] >= CALM_WATCH, "QC_LABEL"] = "관찰"
q.loc[q["CALM_DIFF"] >= CALM_BAD, "QC_LABEL"] = "관측 의심"

out = q.reset_index()
out["YEAR"], out["MONTH"] = out["YM"].dt.year, out["YM"].dt.month
out = out.drop(columns="YM")
out.insert(0, "STN", stn)
with sqlite3.connect(OUT) as conn:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rose_qc'").fetchone()
    if exists:                                          # 같은 지점을 다시 만들어도 중복되지 않게
        conn.execute("DELETE FROM rose_qc WHERE STN=?", (stn,))
    out.to_sql("rose_qc", conn, if_exists="append", index=False)
    conn.commit()

print(f"지점 {stn}: {len(out)}개월 저장 (rose_qc)")
print("\n[QC 라벨별 개월 수]")
print(out["QC_LABEL"].value_counts().to_string())
print("\n[관측 의심·관찰 달]")
print(out[out["QC_LABEL"] != "정상"][
    ["YEAR", "MONTH", "CALM_PCT", "BASE_CALM_PCT", "CALM_DIFF", "SUSPECT_HOURS", "QC_LABEL"]
].to_string(index=False))
print("\n[풍향 고착 후보 달(풍속 3.4 이상, 같은 방위 24시간 이상)]")
print(out[out["STUCK_RUNS"] > 0][["YEAR", "MONTH", "STUCK_RUNS", "STUCK_MAX_HOURS"]].to_string(index=False))
print("\n[자료 부족 달(유효 시간 95% 미만)]")
print(out[out["DATA_LOW"] == 1][["YEAR", "MONTH", "VALID_HOURS", "EXPECTED_HOURS", "VALID_PCT"]].to_string(index=False))
