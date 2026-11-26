"""적재된 asos_hourly를 물리 규칙으로 정리해 asos_hourly_clean 테이블을 만든다."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/kma.db")
SRC, DST = "asos_hourly", "asos_hourly_clean"
TEMP_COLS = ["TA", "TD", "TS", "TE_005", "TE_01", "TE_02", "TE_03"]
CORE = ["TM", "STN", "WD", "WS", "PA", "PS", "PR", "TA", "TD", "HM", "PV",
        "RN", "RN_STATUS", "WS_SUSPECT", "CA_TOT", "VS", "SS", "SI",
        "TS", "TE_005", "TE_01", "TE_02", "TE_03"]


def dewpoint(ta, hm):
    """기온(℃)과 상대습도(%)로 이슬점(℃)을 계산한다 (Magnus 식)."""
    hm = hm.where(hm > 0)
    g = np.log(hm / 100) + 17.62 * ta / (243.12 + ta)
    return 243.12 * g / (17.62 - g)


def mark_ws_persistence(df, min_run=8):
    """R6: 풍속 0이 min_run시간 이상 연속된 시각을 관측 의심(1)으로 표시한다. 값은 유지."""
    df = df.sort_values(["STN", "TM"]).reset_index(drop=True)
    t = pd.to_datetime(df["TM"])
    zero = df["WS"] == 0
    new_run = (
        (zero != zero.shift())
        | (df["STN"] != df["STN"].shift())
        | (t.diff() != pd.Timedelta(hours=1))   # 빠진 시각 앞뒤는 이어 붙이지 않음
    )
    run_len = zero.groupby(new_run.cumsum()).transform("sum")
    df["WS_SUSPECT"] = (zero & (run_len >= min_run)).astype(int)
    return df


def clean(df):
    df = df.copy()
    log = {}

    # R1: -99 계열은 결측 코드 (기온류가 -90℃ 이하일 수 없음)
    for c in TEMP_COLS:
        m = df[c] <= -90
        log[f"R1  {c}: -99 계열 -> 결측"] = int(m.sum())
        df.loc[m, c] = np.nan

    # R2: 이슬점 -9.0은 기온·습도로 계산한 값과 맞으면 실제값, 아니면 결측
    m = df["TD"] == -9.0
    diff = (df["TD"] - dewpoint(df["TA"], df["HM"])).abs()
    real = m & (diff <= 1.0)
    log["R2  TD -9.0 중 실제값(계산 이슬점과 1도 이내, 유지)"] = int(real.sum())
    log["R2  TD -9.0 중 결측 처리"] = int((m & ~real).sum())
    df.loc[m & ~real, "TD"] = np.nan

    # R3: 지면온도 -9.0인데 기온이 0도 이상이면 결측
    m = (df["TS"] == -9.0) & (df["TA"] >= 0)
    log["R3  TS -9.0 (기온 0도 이상) -> 결측"] = int(m.sum())
    df.loc[m, "TS"] = np.nan

    # R4: 기압변화량(PR)은 3시간 간격 관측이라 나머지 시각은 -9.0(결측).
    #     PT가 있는데 -9.0인 8행도 현지기압 3시간 변화(-0.2~-1.8hPa)와 맞지 않아 결측으로 본다
    m = df["PR"] == -9.0
    log["R4  PR -9.0 -> 결측"] = int(m.sum())
    df.loc[m, "PR"] = np.nan

    # R5: 강수. IR=3(무강수)이면 RN을 0으로 채우고 상태를 기록
    log["참고  IR=3(무강수)인데 RN>0인 행"] = int(((df["IR"] == 3) & (df["RN"] > 0)).sum())
    log["참고  IR=4인데 RN 값이 있는 행"] = int(((df["IR"] == 4) & df["RN"].notna()).sum())
    log["참고  그중 RN>0인 행"] = int(((df["IR"] == 4) & (df["RN"] > 0)).sum())
    df["RN_STATUS"] = np.where(
        df["RN"].notna(), "관측", np.where(df["IR"] == 3, "무강수", "불명")
    )
    df.loc[df["RN_STATUS"] == "무강수", "RN"] = 0.0

    # R6: 지속성 검사. 풍속 0이 8시간 이상 이어지면 관측 의심으로 표시(값은 유지)
    df = mark_ws_persistence(df)
    log["R6  풍속 0 연속 8시간 이상 -> 관측 의심 표시(시간 수)"] = int(df["WS_SUSPECT"].sum())
    return df, log


def main():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {SRC}", conn)
        out, log = clean(df)
        out = out[CORE]
        out.to_sql(DST, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{DST} ON {DST} (STN, TM)")
        conn.commit()

    print(f"{DST}: {len(out)}행 저장\n")
    print("=== 규칙별 처리 건수 ===")
    for k, v in log.items():
        print(f"{k}: {v}")

    print("\n=== RN_STATUS ===")
    print(out["RN_STATUS"].value_counts().to_string())

    print("\n=== 정리 후 결측 비율(%) ===")
    miss = (out.isna().mean() * 100).round(1)
    print(miss[miss > 0].sort_values(ascending=False).to_string())

    print("\n=== 정리 후에도 값이 정확히 -9.0인 개수 (기온류) ===")
    print({c: int((out[c] == -9.0).sum()) for c in TEMP_COLS})
    print(f"\n기온(TA) 범위: {out['TA'].min()} ~ {out['TA'].max()}")


if __name__ == "__main__":
    main()
