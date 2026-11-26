"""통합 보고서: 지점·연도를 주면 자료 품질, 작업 이력, 방위 계단, 회전 보정, 월별 특이점을 한 번에 정리한다.
사용법: python src/report.py 159 2026 [--refs 155 255 152] [--after YYYY-MM | --no-correct] [--refresh]
결과: 화면 출력 + reports/<지점>_<연도>.md, reports/<지점>_<연도>.json
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import wr_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("stn", type=int)
ap.add_argument("year", type=int)
ap.add_argument("--refs", nargs="*", type=int, default=None, help="비교 기준 지점(기본: DB에 있는 다른 지점 전부)")
ap.add_argument("--after", default=None, help="이 달(YYYY-MM)부터 방위 보정")
ap.add_argument("--no-correct", action="store_true", help="방위 보정을 하지 않음")
ap.add_argument("--refresh", action="store_true", help="월별 집계와 QC를 다시 만듦")
ap.add_argument("--maint", default="config/maintenance.csv", help="작업 이력 파일")
args = ap.parse_args()
stn, year = args.stn, args.year

# 0) 준비
if L.load_hourly(stn, f"{year}-01-01 00:00:00", f"{year + 1}-01-01 00:00:00").empty:
    sys.exit(f"지점 {stn}의 {year}년 시간자료가 없습니다. collector로 먼저 받으세요.")
if args.refresh or not L.has_monthly(stn):
    for script in ("src/build_monthly.py", "src/qc_wind.py"):
        subprocess.run([sys.executable, script, str(stn)], check=True)
p, lab = L.load_month_matrix(stn)
months = [m for m in range(1, 13) if (year, m) in p.index]

# 1) 자료 품질
qrows = []
for m in months:
    r = lab.loc[(year, m)]
    tags = []
    if r["QC_LABEL"] != "정상":
        tags.append(r["QC_LABEL"])
    if r["CALM_DIFF"] <= -5:
        tags.append("정온 감소")
    if r["DATA_LOW"] == 1:
        tags.append("자료 부족")
    if r["STUCK_RUNS"] > 0:
        tags.append(f"고착 후보 {int(r['STUCK_RUNS'])}건")
    if r["EXPECTED_HOURS"] < pd.Period(f"{year}-{m:02d}").days_in_month * 24:
        tags.append("미완 월")
    qrows.append({"월": m, "유효시간%": round(float(r["VALID_PCT"]), 1), "정온%": float(r["CALM_PCT"]),
                  "기준정온%": float(r["BASE_CALM_PCT"]), "정온차%p": float(r["CALM_DIFF"]),
                  "R6(시간)": int(r["SUSPECT_HOURS"]), "표시": ", ".join(tags) or "-"})
qnote = {q["월"]: q["표시"] for q in qrows}

# 2) 작업 이력
maint = []
if Path(args.maint).exists():
    md = pd.read_csv(args.maint, dtype=str)
    maint = md[md["stn"].astype(int) == stn].to_dict("records")

# 3) 방위 계단 탐지
refs = args.refs if args.refs is not None else [s for s in L.available_stations() if s != stn]
start = pd.Timestamp(f"{year - 1}-07-01")
start += pd.Timedelta(days=(7 - start.weekday()) % 7)          # 다음 월요일부터
end = pd.Timestamp(f"{year + 1}-01-01")
wk = L.weekly_yoy(stn, refs, start, end) if refs else pd.DataFrame()
step = L.detect_step(wk) if len(wk) >= 10 else None
matched = []
if step:
    for e in maint:
        if pd.Timestamp(e["start"]) - pd.Timedelta(days=7) <= step["date"] <= pd.Timestamp(e["end"]) + pd.Timedelta(days=7):
            matched.append(e)

# 4) 회전 보정
after = None
if args.after:
    after = tuple(int(x) for x in args.after.split("-"))
elif step and step["is_step"] and matched:          # 작업 이력과 맞을 때만 자동 보정
    d = step["date"] + pd.Timedelta(days=3)
    if d.day > 10:
        d = d + pd.offsets.MonthBegin(1)
    after = (d.year, d.month)
if args.no_correct:
    after = None
rot = {}
if after:
    for m in months:
        if (year, m) >= after:
            f = L.month_rotation(p, lab, year, m)
            if f is not None:
                rot[m] = f
f_corr = float(np.median(list(rot.values()))) if rot else 0.0
corr_deg = -f_corr * 22.5

# 5) 월별 판정
mrows = []
for m in months:
    good = L.good_years(p, lab, year, m)
    if len(good) < L.MIN_BASE:
        continue
    B = np.array([p.loc[(y, m)].values for y in good])
    t = p.loc[(year, m)].values
    raw = L.evaluate(t, B)
    fixed = bool(rot) and after is not None and (year, m) >= after
    cor = L.evaluate(L.rotate_cells(t, f_corr), B) if fixed else raw
    note = [qnote[m]] if qnote[m] != "-" else []
    if fixed:
        note.append("보정 적용")
    mrows.append({"월": m, "순위(원)": f"{raw['rank']}/{raw['n']}", "거리비(원)": round(raw["ratio"], 2),
                  "판정(원)": raw["verdict"], "순위(보정)": f"{cor['rank']}/{cor['n']}",
                  "거리비(보정)": round(cor["ratio"], 2), "판정(보정)": cor["verdict"],
                  "원인 칸": cor["causes"], "비고": ", ".join(note) or "-"})


# 6) 요약과 출력
def mlist(rows, key, values):
    ms = [str(r["월"]) for r in rows if r[key] in values]
    return ", ".join(ms) + "월" if ms else "없음"


def qlist(tag):
    ms = [str(q["월"]) for q in qrows if tag in q["표시"]]
    return ", ".join(ms) + "월" if ms else "없음"


summary = [f"원자료 판정: 특이 {mlist(mrows, '판정(원)', ['특이'])}, 주목 {mlist(mrows, '판정(원)', ['주목'])}"]
if step:
    if step["is_step"] and matched:
        kind = "계단형(작업 이력 일치, 자동 보정)"
    elif step["is_step"]:
        kind = "계단 후보(작업 이력 없음, 자동 보정 안 함)"
    else:
        kind = "뚜렷한 계단 아님"
    ev = "; ".join(f"{e['start']}~{e['end']} {e['kind']}" for e in matched) or "일치하는 작업 이력 없음"
    summary.append(f"방위 계단: {step['date']:%Y-%m-%d} 주, 변화 {step['delta']:+.0f}° (t={step['t']:.1f}) → {kind}, {ev}")
if after and rot:
    summary.append(f"방위 보정({after[0]}-{after[1]:02d}부터, {corr_deg:+.1f}°) 후 판정: "
                   f"특이 {mlist(mrows, '판정(보정)', ['특이'])}, 주목 {mlist(mrows, '판정(보정)', ['주목'])}")
summary.append(f"자료 품질: 관측 의심 {qlist('관측 의심')}, 관찰 {qlist('관찰')}, 정온 감소 {qlist('정온 감소')}")
repl = [e for e in maint if str(e["kind"]).startswith("장비 교체") and year - L.N_BASE <= int(e["start"][:4]) < year]
if repl:
    summary.append("주의: 기준 기간에 장비 교체가 있음(" + ", ".join(e["start"] for e in repl)
                   + ") → 정온·방위 비교에 교체 전후 장비 특성 차이가 섞임")


def md_table(rows):
    if not rows:
        return "(없음)"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


md = [f"# 바람장미 특이점 보고서: 지점 {stn}, {year}년",
      f"생성 {datetime.now():%Y-%m-%d %H:%M}, 기준: 직전 {L.N_BASE}년 같은 달(관측 의심·자료 부족 달 제외)", "",
      "## 요약", *[f"- {s}" for s in summary], "",
      "## 1. 자료 품질", md_table(qrows), "",
      "## 2. 작업 이력", md_table(maint) if maint else "(기록 없음)", "",
      f"## 3. 방위 계단 탐지 (기준 지점: {', '.join(map(str, refs)) or '없음'})"]
if step:
    md.append(f"- 가장 뚜렷한 변화: {step['date']:%Y-%m-%d} 주, 작년 같은 주 대비 변화 평균 "
              f"{step['before']:+.0f}° → {step['after']:+.0f}° (차이 {step['delta']:+.0f}°, t={step['t']:.1f})")
    k = step["k"]
    ex = wk.iloc[max(0, k - 6): k + 6]
    md.append(md_table([{"주 시작": f"{r.week_start:%Y-%m-%d}", "변화(°)": round(r.change), "N": int(r.N)}
                        for r in ex.itertuples()]))
else:
    md.append("(비교할 주별 자료가 부족합니다)")
md += ["", "## 4. 방위 보정"]
if after and rot:
    md.append(f"- {after[0]}-{after[1]:02d}부터 보정, 월별 추정 회전(°): "
              + ", ".join(f"{m}월 {-f * 22.5:+.1f}" for m, f in rot.items())
              + f" → 공통 보정량 {corr_deg:+.1f}° (+ = 관측이 시계 방향으로 돌아가 있음)")
else:
    md.append("- 보정하지 않음")
md += ["", "## 5. 월별 판정", md_table(mrows), "",
       "판정 기준: 순위 1 = 기준 연도 중 가장 먼 해보다 더 멂, 거리비 = 올해 거리 / 기준 연도 최대 거리, "
       "특이 = 순위 1이면서 원인 칸 있음, 주목 = 순위 2 이내."]
text = "\n".join(md)
print(text)

out = Path("reports")
out.mkdir(exist_ok=True)
(out / f"{stn}_{year}.md").write_text(text + "\n", encoding="utf-8")
data = {"stn": stn, "year": year, "refs": refs, "summary": summary, "qc": qrows, "maintenance": maint,
        "step": step, "matched": matched,
        "weekly": wk.assign(week_start=wk["week_start"].dt.strftime("%Y-%m-%d")).to_dict("records") if len(wk) else [],
        "correction": {"after": f"{after[0]}-{after[1]:02d}" if after else None,
                       "monthly_deg": {m: -f * 22.5 for m, f in rot.items()}, "deg": corr_deg if rot else 0.0},
        "months": mrows}
(out / f"{stn}_{year}.json").write_text(
    json.dumps(data, ensure_ascii=False, indent=2,
               default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print(f"\n저장: reports/{stn}_{year}.md, reports/{stn}_{year}.json")
