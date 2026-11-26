"""지점·기간을 정해 포털 바람장미 표와 같은 기준의 집계표를 출력한다.
사용법: python src/run_table.py [지점번호] [시작월] [종료월]
예:     python src/run_table.py 159 2025-01 2025-12
"""
import sys

import pandas as pd

from windrose_core import classify, load_hourly, rose_table

stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
p1 = pd.Period(sys.argv[2] if len(sys.argv) > 2 else "2025-01", "M")
p2 = pd.Period(sys.argv[3] if len(sys.argv) > 3 else "2025-12", "M")
start, end = p1.start_time, (p2 + 1).start_time

df = load_hourly(stn, start, end)
expected = int((end - start) / pd.Timedelta(hours=1))
print(f"지점 {stn}, {p1} ~ {p2}")
print(f"기대 {expected}시간, DB 행 {len(df)}, 풍속 결측 {int(df['WS'].isna().sum())}")

d = classify(df.dropna(subset=["WS"]))
print("\n[점검] 정온(풍속 0.4 이하)과 풍향 0의 일치 여부")
print(f"  풍속 <= 0.4 이면서 풍향 != 0: {int(((d['SPD'] == 0) & (d['WD'] != 0)).sum())}행")
print(f"  풍속 > 0.4 이면서 풍향 == 0: {int(((d['SPD'] > 0) & (d['WD'] == 0)).sum())}행")
print(f"  변환표에 없는 풍향 코드: {int((d['WD'].notna() & (d['WD'] != 0) & d['DIR'].isna()).sum())}행")

pct, total, unknown = rose_table(df)
print(f"\n[바람장미 표] 분모 {total}시간, 방향 불명 {unknown}시간 (단위 %, 화면 폭 때문에 방위를 세로로 놓음)")
pd.set_option("display.width", 200)
print(pct.T.round(1).to_string())
