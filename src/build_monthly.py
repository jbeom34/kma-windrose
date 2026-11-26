"""선택 지점의 월별 바람장미 집계를 data/windrose.db에 저장한다.
사용법: python src/build_monthly.py [지점번호]
저장 표: rose_cells(월·풍속계급·방위별 시간 수), rose_meta(월별 유효 시간·관측 의심 시간)
방위 번호: 0~15(북~북북서), -1=정온, -2=방향 불명
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from windrose_core import classify, load_hourly

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
OUT = Path("data/windrose.db")

df = load_hourly(stn, "2015-01-01 00:00:00", "2100-01-01 00:00:00")
df["TM"] = pd.to_datetime(df["TM"])
df["YEAR"], df["MONTH"] = df["TM"].dt.year, df["TM"].dt.month

valid = classify(df.dropna(subset=["WS"]))
valid["DIR"] = valid["DIR"].fillna(-2)
valid.loc[valid["SPD"] == 0, "DIR"] = -1
valid["DIR"] = valid["DIR"].astype(int)

cells = (valid.groupby(["YEAR", "MONTH", "SPD", "DIR"]).size()
         .rename("HOURS").reset_index())
cells.insert(0, "STN", stn)

meta = df.groupby(["YEAR", "MONTH"]).agg(
    ROWS=("TM", "size"), VALID_HOURS=("WS", "count"),
    SUSPECT_HOURS=("WS_SUSPECT", "sum")).reset_index()
meta["SUSPECT_HOURS"] = meta["SUSPECT_HOURS"].astype(int)
first_day = pd.to_datetime(dict(year=meta["YEAR"], month=meta["MONTH"], day=1))
meta["EXPECTED_HOURS"] = first_day.dt.days_in_month * 24
last = df["TM"].max()                      # 진행 중인 달은 마지막 자료 시각까지만 기대
mask = (meta["YEAR"] == last.year) & (meta["MONTH"] == last.month)
meta.loc[mask, "EXPECTED_HOURS"] = int((last - last.replace(day=1, hour=0)) / pd.Timedelta(hours=1)) + 1
meta.insert(0, "STN", stn)

OUT.parent.mkdir(exist_ok=True)
with sqlite3.connect(OUT) as conn:
    for name, frame in (("rose_cells", cells), ("rose_meta", meta)):
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        if exists:                         # 같은 지점을 다시 만들어도 중복되지 않게 먼저 지운다
            conn.execute(f"DELETE FROM {name} WHERE STN=?", (stn,))
        frame.to_sql(name, conn, if_exists="append", index=False)
    conn.commit()

chk = (cells.groupby(["YEAR", "MONTH"])["HOURS"].sum().reset_index()
       .merge(meta, on=["YEAR", "MONTH"]))
bad = chk[chk["HOURS"] != chk["VALID_HOURS"]]
print(f"지점 {stn}: 집계 {len(cells)}칸, {len(meta)}개월 저장 ({OUT})")
print(f"월별 시간 합계가 유효 시간과 다른 달: {len(bad)}개 (0이어야 정상)")
low = meta[meta["VALID_HOURS"] < meta["EXPECTED_HOURS"] * 0.98]
print("\n유효 시간이 기대의 98% 미만인 달:")
print(low.to_string(index=False) if len(low) else "  없음")
print("\n관측 의심(R6) 시간이 있는 달:")
sus = meta[meta["SUSPECT_HOURS"] > 0]
print(sus[["YEAR", "MONTH", "SUSPECT_HOURS"]].to_string(index=False) if len(sus) else "  없음")
