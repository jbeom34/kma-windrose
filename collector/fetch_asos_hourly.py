"""ASOS 시간자료를 연도별 파일로 받아 data/raw에 저장한다.

사용법: python collector/fetch_asos_hourly.py [지점번호] [시작연도] [종료연도]
예:     python collector/fetch_asos_hourly.py 159 2015 2025
이미 받은 연도는 건너뛰므로, 중간에 멈춰도 다시 실행하면 이어서 받는다.
"""
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("KMA_AUTH_KEY")
URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
CHUNK_DAYS = 30            # 한 번에 요청 가능한 최대 기간
OUT_DIR = Path("data/raw")


def chunks(start, end, days):
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=days - 1), end)
        yield cur, last
        cur = last + timedelta(days=1)


def fetch(stn, tm1, tm2, tries=5):
    """성공하면 응답 줄 목록, 실패하면 None. (오류 시 인증키가 출력되지 않게 주의)"""
    params = {"tm1": tm1, "tm2": tm2, "stn": stn, "help": 1, "authKey": KEY}
    for n in range(1, tries + 1):
        try:
            r = requests.get(URL, params=params, timeout=90)
            if r.status_code == 200:
                lines = r.text.splitlines()
                if any(x.strip() and not x.startswith("#") for x in lines):
                    return lines
                err = f"자료 없음: {r.text[:200]}"
            else:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            err = type(e).__name__
        print(f"  재시도 {n}/{tries}: {err}")
        time.sleep(3 * n)
    return None


def fetch_year(stn, year):
    path = OUT_DIR / f"asos_hourly_{stn}_{year}0101_{year}1231.txt"
    if path.exists():
        print(f"{year}: 이미 있음, 건너뜀")
        return True

    header, rows = [], []
    for a, b in chunks(date(year, 1, 1), date(year, 12, 31), CHUNK_DAYS):
        lines = fetch(stn, f"{a:%Y%m%d}0000", f"{b:%Y%m%d}2300")
        if lines is None:
            print(f"{year}: {a}~{b} 실패, 이 연도는 저장하지 않고 중단합니다.")
            return False
        if not header:
            header = [x for x in lines if x.startswith("#")]
        rows += [x for x in lines if x.strip() and not x.startswith("#")]
        time.sleep(0.5)  # 서버 부담을 줄이기 위한 대기

    expected = (date(year, 12, 31) - date(year, 1, 1)).days * 24 + 24
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    note = "" if len(rows) == expected else f"  [주의] 예상 {expected}행과 다름"
    print(f"{year}: {len(rows)}행 저장{note}")
    return True


def main():
    if not KEY:
        raise SystemExit(".env 파일에 KMA_AUTH_KEY가 없습니다.")
    last_full = date.today().year - 1
    stn = int(sys.argv[1]) if len(sys.argv) > 1 else 159
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    y2 = int(sys.argv[3]) if len(sys.argv) > 3 else last_full
    if y2 > last_full:
        raise SystemExit(f"종료연도는 {last_full}년까지만 가능합니다(진행 중인 연도는 제외).")
    for year in range(y1, y2 + 1):
        if not fetch_year(stn, year):
            break


if __name__ == "__main__":
    main()
