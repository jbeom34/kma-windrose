"""바람장미 계산의 핵심 규칙: 방위 코드 변환, 풍속 계급, 집계표 만들기."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path.home() / "kma-anomaly" / "data" / "kma.db"

# DB의 풍향 코드(16방위를 36 눈금으로 환산한 값) -> 16방위 번호(0=북, 1=북북동, ..., 15=북북서)
CODE_TO_IDX = {36: 0, 2: 1, 5: 2, 7: 3, 9: 4, 11: 5, 14: 6, 16: 7,
               18: 8, 20: 9, 23: 10, 25: 11, 27: 12, 29: 13, 32: 14, 34: 15}
DIR_NAMES = ["북", "북북동", "북동", "동북동", "동", "동남동", "남동", "남남동",
             "남", "남남서", "남서", "서남서", "서", "서북서", "북서", "북북서"]
SPEED_LABELS = ["~0.4(정온)", "0.5~3.3", "3.4~7.9", "8.0~13.8", "13.9~"]
SPEED_EDGES = [0.5, 3.4, 8.0, 13.9]   # 각 계급의 시작값


def load_hourly(stn, start, end):
    """kma.db에서 [start, end) 기간의 시간자료(TM, WD, WS)를 읽기 전용으로 가져온다."""
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        return pd.read_sql(
            "SELECT TM, WD, WS FROM asos_hourly_clean WHERE STN=? AND TM>=? AND TM<?",
            conn, params=(stn, str(start), str(end)),
        )


def classify(df):
    """풍속 계급 번호(SPD)와 16방위 번호(DIR)를 붙인다. 정온·결측의 DIR은 비어 있다."""
    df = df.copy()
    df["SPD"] = np.digitize(df["WS"].round(1), SPEED_EDGES)
    df["DIR"] = df["WD"].map(lambda c: CODE_TO_IDX.get(int(c), np.nan) if pd.notna(c) else np.nan)
    return df


def rose_table(df):
    """풍속 계급 x 방위 빈도표(%)를 만든다. 풍속 결측 시간은 분모에서 뺀다."""
    d = classify(df.dropna(subset=["WS"]))
    total = len(d)
    tab = pd.DataFrame(0.0, index=SPEED_LABELS, columns=["정온"] + DIR_NAMES)
    tab.loc[SPEED_LABELS[0], "정온"] = (d["SPD"] == 0).sum()
    for (s, k), n in d[d["SPD"] > 0].groupby(["SPD", "DIR"]).size().items():
        tab.loc[SPEED_LABELS[int(s)], DIR_NAMES[int(k)]] = n
    unknown = int(((d["SPD"] > 0) & d["DIR"].isna()).sum())
    pct = tab / total * 100
    pct["합계"] = pct.sum(axis=1)
    pct.loc["합계"] = pct.sum(axis=0)
    return pct, total, unknown
