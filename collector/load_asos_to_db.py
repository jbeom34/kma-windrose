"""data/raw의 ASOS 시간자료 텍스트를 읽어 결측을 정리하고 SQLite에 저장한다."""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/kma.db")
TABLE = "asos_hourly"

# 파일 헤더의 컬럼 순서(46개)와 같은 순서로 붙인 이름
COLS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT",
    "PR", "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT",
    "SD_HR3", "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID",
    "CH_MIN", "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD",
    "TS", "TE_005", "TE_01", "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]
TEXT_COLS = ["WW", "CT"]                      # 결측이 '-'인 문자 컬럼
KEEP_RAW = ["TA", "TD", "PR", "TS", "TE_005", "TE_01", "TE_02", "TE_03"]  # 음수 가능 -> -9.0 자동변환 안 함
HEAD_N, TAIL_N = 24, 21                       # WW 앞 24개, 뒤 21개 컬럼은 고정


def parse_line(line):
    """한 줄을 46개 값으로 나눈다. WW는 공백이 있어도 가운데 부분을 합쳐서 처리."""
    t = line.split()
    if len(t) < len(COLS):
        return None
    ww = " ".join(t[HEAD_N:len(t) - TAIL_N])
    return t[:HEAD_N] + [ww] + t[-TAIL_N:]


def load_file(path):
    rows, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        row = parse_line(line)
        if row is None:
            bad += 1
        else:
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)

    # 시각: 202401010000 -> 2024-01-01 00:00:00 (한국시간 그대로)
    df["TM"] = pd.to_datetime(df["TM"], format="%Y%m%d%H%M").dt.strftime("%Y-%m-%d %H:%M:%S")

    num_cols = [c for c in COLS if c not in TEXT_COLS and c != "TM"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    to_mask = [c for c in num_cols if c not in KEEP_RAW and c != "STN"]
    df[to_mask] = df[to_mask].mask(df[to_mask] == -9)       # -9, -9.0, -9.00 -> 결측
    df[TEXT_COLS] = df[TEXT_COLS].mask(df[TEXT_COLS] == "-")  # '-' -> 결측
    return df, bad


def save(df, conn):
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)
    ).fetchone()
    if exists:  # 같은 지점·기간을 다시 넣어도 중복되지 않게 먼저 지운다
        for stn in df["STN"].unique():
            sub = df[df["STN"] == stn]
            conn.execute(
                f"DELETE FROM {TABLE} WHERE STN=? AND TM BETWEEN ? AND ?",
                (int(stn), sub["TM"].min(), sub["TM"].max()),
            )
    df.to_sql(TABLE, conn, if_exists="append", index=False)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{TABLE} ON {TABLE} (STN, TM)")
    conn.commit()


def report(conn):
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    print("\n=== 검증 ===")
    print(f"총 행 수: {len(df)}")
    print(f"지점: {sorted(df['STN'].unique().tolist())}")
    print(f"기간: {df['TM'].min()} ~ {df['TM'].max()}")

    miss = (df.isna().mean() * 100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    print("\n결측 비율(%) - 결측이 있는 컬럼만:")
    print(miss.to_string())

    print("\n원본 유지 컬럼 중 값이 정확히 -9.0인 행 수 (결측인지 실제값인지 확인 필요):")
    for c in KEEP_RAW:
        print(f"  {c}: {(df[c] == -9.0).sum()}")
    print(f"\n기온(TA) 범위: {df['TA'].min()} ~ {df['TA'].max()}")


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("data/raw").glob("*.txt"))
    if not paths:
        raise SystemExit("data/raw에 읽을 .txt 파일이 없습니다.")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for p in paths:
            df, bad = load_file(p)
            save(df, conn)
            print(f"{p.name}: {len(df)}행 저장 (형식이 다른 줄 {bad}개 건너뜀)")
        report(conn)


if __name__ == "__main__":
    main()
