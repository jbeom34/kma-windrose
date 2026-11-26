"""바람장미 특이점 웹 서버.
실행(저장소 최상위 폴더에서): uvicorn app.main:app --port 8001
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import wr_lib as L  # noqa: E402

NAMES = {159: "부산", 155: "창원", 255: "북창원", 152: "울산"}
app = FastAPI(title="바람장미 특이점")


def report_path(stn, year):
    return ROOT / "reports" / f"{stn}_{year}.json"


def shape(t):
    """65칸 비율 -> 정온 비율(%)과 풍속 계급별 16방위 비율(%)."""
    return {"calm": round(float(t[0]) * 100, 2),
            "speeds": [[round(float(t[1 + k * 4 + s]) * 100, 2) for k in range(16)] for s in range(4)]}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/stations")
def stations():
    return [{"stn": s, "name": NAMES.get(s, str(s))} for s in L.available_stations()]


@app.get("/api/report/{stn}/{year}")
def report(stn: int, year: int, refresh: bool = False):
    path = report_path(stn, year)
    if refresh or not path.exists():
        r = subprocess.run([sys.executable, "src/report.py", str(stn), str(year)],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise HTTPException(500, f"보고서 생성 실패: {(r.stderr or r.stdout)[-500:]}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/rose/{stn}/{year}/{month}")
def rose(stn: int, year: int, month: int):
    p, lab = L.load_month_matrix(stn)
    if (year, month) not in p.index:
        raise HTTPException(404, "해당 달 자료가 없습니다")
    good = L.good_years(p, lab, year, month)
    t = p.loc[(year, month)].values
    out = {"stn": stn, "year": year, "month": month, "base_years": good, "dirs": L.DIR_NAMES,
           "raw": shape(t), "corrected": None, "correction_deg": 0.0,
           "base": shape(np.array([p.loc[(y, month)].values for y in good]).mean(axis=0)) if good else None}
    path = report_path(stn, year)
    if path.exists():
        c = json.loads(path.read_text(encoding="utf-8")).get("correction") or {}
        if c.get("after") and c.get("deg"):
            after = tuple(int(x) for x in c["after"].split("-"))
            if (year, month) >= after:
                out["corrected"] = shape(L.rotate_cells(t, -c["deg"] / 22.5))
                out["correction_deg"] = c["deg"]
    return out
