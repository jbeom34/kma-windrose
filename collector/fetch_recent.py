"""올해 1월 1일부터 어제까지의 ASOS 시간자료를 받아 data/raw에 저장한다(실행할 때마다 덮어씀).

사용법: python collector/fetch_recent.py [지점번호]
"""
import sys
import time
from datetime import date, timedelta

from fetch_asos_hourly import CHUNK_DAYS, KEY, OUT_DIR, chunks, fetch


def main():
    if not KEY:
        raise SystemExit(".env 파일에 KMA_AUTH_KEY가 없습니다.")
    stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
    today = date.today()
    start = date(today.year, 1, 1)
    end = today - timedelta(days=1)
    if end < start:
        raise SystemExit("받을 최근 자료가 없습니다(새해 첫날).")

    header, rows = [], []
    for a, b in chunks(start, end, CHUNK_DAYS):
        lines = fetch(stn, f"{a:%Y%m%d}0000", f"{b:%Y%m%d}2300")
        if lines is None:
            raise SystemExit(f"{a}~{b} 구간을 받지 못해 중단합니다. 기존 파일은 그대로 둡니다.")
        if not header:
            header = [x for x in lines if x.startswith("#")]
        rows += [x for x in lines if x.strip() and not x.startswith("#")]
        time.sleep(0.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"asos_hourly_{stn}_recent.txt"
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    expected = ((end - start).days + 1) * 24
    print(f"{start} ~ {end}: {len(rows)}행 저장 (예상 {expected}행) -> {path}")


if __name__ == "__main__":
    main()
