"""10-2 카메라 위치 보정 (카메라를 놓은 뒤 1번): 줄자로 잰 기준점을 화면에서 클릭 → solvePnP
사용법 (렌즈 보정을 먼저 끝낸 뒤, 카메라 하나씩)
  python calib_extrinsics.py --src 0 --name cam0
  python calib_extrinsics.py --src vigi:stream2 --name cam1
  → 브라우저 http://127.0.0.1:8002
준비: calib/world_points.json 에 기준점 좌표(m)를 적는다. 좌표계는 stereo.py와 같다
  (x = 칠판과 평행 오른쪽 +, y = 교실 뒤쪽 +, z = 위 +). "check": true 인 점은 계산에 안 쓰고 검증에만 쓴다.
진행: [화면 멈춤] → 목록에서 점 고르기 → 멈춘 화면에서 그 점 클릭 → ... → [위치 계산]
결과: calib/<name>.npz (K, dist, R, t — 앱이 읽는 파일), calib/<name>_clicks.json
      다른 카메라도 끝나 있으면 두 카메라가 같이 클릭한 점을 삼각측량해 3D 오차를 보여 준다 (10-2 검증).
"""
import argparse
import json
import os
import threading

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

import camio
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--res", default="640x480")
ap.add_argument("--focus", type=int, default=0)
ap.add_argument("--world", default="calib/world_points.json")
ap.add_argument("--other", default=None, help="삼각측량 검증에 쓸 다른 카메라 이름 (기본: cam0↔cam1)")
ap.add_argument("--port", type=int, default=8002)
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()
OTHER = args.other or ("cam1" if args.name.endswith("0") else "cam0")

intr = np.load(f"calib/{args.name}_intr.npz")
K, DIST = intr["K"], intr["dist"]
WORLD = {p["id"]: p for p in S.load_world(args.world)}
src = camio.Source(args.src, tuple(map(int, args.res.split("x"))), focus=args.focus)
first = camio.wait_frame(src)
SIZE = (first.shape[1], first.shape[0])
INTR_SIZE = tuple(int(v) for v in (intr["size"] if "size" in intr else intr["res"]))   # 키트 형식(res)도 읽음
if INTR_SIZE != SIZE:
    raise SystemExit(f"렌즈 보정 해상도 {INTR_SIZE} ≠ 지금 {SIZE}. 같은 --res로 열거나 렌즈 보정을 다시 하세요")
if "src" in intr and str(intr["src"]) != args.src:
    print(f"주의: 렌즈 보정 때 입력 {intr['src']} ≠ 지금 {args.src}", flush=True)

clicks_path = f"calib/{args.name}_clicks.json"
clicks = json.load(open(clicks_path)) if os.path.exists(clicks_path) else {}
frozen = {"img": None}
state = {"result": None, "pose": None}
lock = threading.Lock()


def draw(img):
    vis = img.copy()
    pose = state["pose"]
    for pid, (u, v) in clicks.items():
        chk = WORLD.get(pid, {}).get("check")
        col = (255, 160, 0) if chk else (0, 255, 255)
        cv2.drawMarker(vis, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 14, 1)
        cv2.putText(vis, pid, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if pose is not None:                     # 계산 뒤: 모든 기준점을 다시 투영해 빨간 원으로 (클릭과 겹치면 정확)
        for pid, p in WORLD.items():
            uv = S.project_distorted(pose, p["xyz"])
            if 0 <= uv[0] < SIZE[0] and 0 <= uv[1] < SIZE[1]:
                cv2.circle(vis, (int(uv[0]), int(uv[1])), 5, (0, 0, 255), 1)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def live_jpeg():
    t, f = src.latest()
    return None if f is None else draw(f)


def solve():
    fit = [(pid, uv) for pid, uv in clicks.items() if pid in WORLD and not WORLD[pid].get("check")]
    if len(fit) < 4:
        return {"error": f"계산용 점이 최소 4개 필요합니다 (지금 {len(fit)}개)"}
    obj = np.float64([WORLD[p]["xyz"] for p, _ in fit])
    img = np.float64([uv for _, uv in fit])
    planar = np.ptp(obj[:, 2]) < 0.05
    if planar:                                        # 모두 바닥(같은 높이)이면 평면 전용 풀이 → 반복 정밀화
        ok, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_IPPE)
    else:
        ok, rv, tv = cv2.solvePnP(obj, img, K, DIST, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return {"error": "solvePnP 실패"}
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, DIST, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    pose = {"K": K, "dist": DIST, "R": R, "t": tv.reshape(3, 1), "P": K @ np.hstack([R, tv.reshape(3, 1)]),
            "C": (-R.T @ tv).ravel(), "Kinv": np.linalg.inv(K)}
    per = {}
    for pid, uv in clicks.items():
        if pid not in WORLD:
            continue
        p = WORLD[pid]
        e_px = float(np.linalg.norm(S.project_distorted(pose, p["xyz"]) - np.float64(uv)))
        X = S.ray_plane(pose, S.undistort(pose, [uv])[0], p["xyz"][2])      # 클릭 → 그 점 높이 평면 위 위치
        e_m = None if X is None else float(np.linalg.norm(X[:2] - np.float64(p["xyz"][:2])))
        per[pid] = {"용도": "검증" if p.get("check") else "계산", "재투영_px": round(e_px, 2),
                    "바닥_오차_cm": None if e_m is None else round(100 * e_m, 1)}
    fit_px = [v["재투영_px"] for v in per.values() if v["용도"] == "계산"]
    chk_cm = [v["바닥_오차_cm"] for v in per.values() if v["용도"] == "검증" and v["바닥_오차_cm"] is not None]
    np.savez(f"calib/{args.name}.npz", K=K, dist=DIST, R=R, t=tv, size=np.array(SIZE), res=np.array(SIZE), src=args.src)
    state["pose"] = pose
    res = {"저장": f"calib/{args.name}.npz", "풀이": "평면(IPPE)" if planar else "SQPnP", "계산용_점": len(fit),
           "카메라_위치_m": np.round(pose["C"], 3).tolist(),
           "계산용_재투영_평균_px": round(float(np.mean(fit_px)), 2),
           "검증점_바닥_오차_cm": chk_cm, "점별": per,
           "판정": "좋음" if np.mean(fit_px) < 2 and (not chk_cm or max(chk_cm) < 10) else
                   "점검 필요 (클릭 위치·줄자 값·점 이름을 확인)"}
    res["두_카메라_검증"] = stereo_check(pose)
    state["result"] = res
    print(json.dumps({k: v for k, v in res.items() if k != "점별"}, ensure_ascii=False), flush=True)
    return res


def stereo_check(pose):
    """다른 카메라 보정이 있으면: 두 카메라가 모두 클릭한 점을 삼각측량해 실제 좌표와 비교 (10-2 3D 검증)"""
    op, oc = f"calib/{OTHER}.npz", f"calib/{OTHER}_clicks.json"
    if not (os.path.exists(op) and os.path.exists(oc)):
        return f"{OTHER} 위치 보정이 아직 없음"
    other, oclicks = S.load_cam(op), json.load(open(oc))
    me = pose
    c0, c1 = (me, other) if args.name < OTHER else (other, me)
    k0, k1 = (clicks, oclicks) if args.name < OTHER else (oclicks, clicks)
    out = {}
    for pid in sorted(set(k0) & set(k1) & set(WORLD)):
        p0, p1 = S.undistort(c0, [k0[pid]])[0], S.undistort(c1, [k1[pid]])[0]
        X, err = S.triangulate(c0, c1, p0, p1)
        out[pid] = {"3D_오차_cm": round(100 * float(np.linalg.norm(X - np.float64(WORLD[pid]["xyz"]))), 1),
                    "재투영_px": round(err, 2), "높이_m": round(float(X[2]), 3)}
    if not out:
        return "두 카메라가 같이 클릭한 점이 없음"
    base = float(np.linalg.norm(c0["C"] - c1["C"]))
    return {"기준선_m": round(base, 2), "점별": out,
            "평균_3D_오차_cm": round(float(np.mean([v["3D_오차_cm"] for v in out.values()])), 1)}


app = FastAPI()


@app.get("/stream")
def stream():
    return StreamingResponse(camio.mjpeg(live_jpeg), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/freeze")
def freeze():
    t, f = src.latest()
    frozen["img"] = f.copy()                  # 메모리에만 둔다 (디스크 저장 안 함)
    return {"ok": True, "w": SIZE[0], "h": SIZE[1]}


@app.get("/frozen.jpg")
def frozen_jpg():
    if frozen["img"] is None:
        return Response(status_code=404)
    return Response(draw(frozen["img"]), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/click")
async def click(req: Request):
    d = await req.json()
    with lock:
        if d.get("remove"):
            clicks.pop(d["id"], None)
        else:
            clicks[d["id"]] = [float(d["u"]), float(d["v"])]
        json.dump(clicks, open(clicks_path, "w"), ensure_ascii=False, indent=1)
    state["pose"] = None
    return {"ok": True, "n": len(clicks)}


@app.get("/state")
def get_state():
    return {"points": [{"id": p, "xyz": WORLD[p]["xyz"], "check": bool(WORLD[p].get("check")),
                        "clicked": p in clicks} for p in WORLD], "result": state["result"], "size": SIZE}


@app.post("/solve")
def do_solve():
    return solve()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>위치 보정 NAME</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:14px;flex-wrap:wrap}
#img{cursor:crosshair;max-width:100%;border-radius:6px}button{font-size:15px;padding:6px 12px;margin:3px 3px 3px 0}
.pt{display:block;width:100%;text-align:left}.sel{outline:2px solid #4af}.done{color:#5d5}.chk{color:#fa0}
pre{background:#1b1b1b;padding:8px;white-space:pre-wrap;font-size:12px;max-width:640px}</style>
<h2>위치 보정 — NAME</h2>
<p>① <button onclick="freeze()">화면 멈춤</button><button onclick="live()">실시간 보기</button>
 ② 오른쪽 목록에서 점 고르기 ③ 멈춘 화면에서 그 점(테이프 표시의 정확한 위치)을 클릭 ④ <button onclick="solve()">위치 계산</button>
 <br><small>노랑 = 계산용, 주황 = 검증용, 빨간 원 = 계산한 위치로 다시 투영한 자리 (클릭과 겹칠수록 정확). 브라우저 확대(Ctrl/⌘ +)로 크게 보고 찍으면 정확합니다.</small></p>
<div class=row><div><img id=img src="/stream" onclick="clk(event)"></div>
<div style="min-width:220px"><div id=pts></div><button onclick="rm()">고른 점 클릭 지우기</button></div></div>
<pre id=res></pre>
<script>
let sel=null,isFrozen=false;
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):null})).json()}
async function freeze(){await post('/freeze');isFrozen=true;refresh()}
function live(){isFrozen=false;document.getElementById('img').src='/stream'}
function refresh(){if(isFrozen)document.getElementById('img').src='/frozen.jpg?'+Date.now()}
async function clk(e){if(!isFrozen){alert('먼저 [화면 멈춤]을 누르세요');return}if(!sel){alert('목록에서 점을 먼저 고르세요');return}
 const im=e.target,r=im.getBoundingClientRect(),u=(e.clientX-r.left)*im.naturalWidth/r.width,v=(e.clientY-r.top)*im.naturalHeight/r.height;
 await post('/click',{id:sel,u:u,v:v});await load();refresh()}
async function rm(){if(sel){await post('/click',{id:sel,remove:true});await load();refresh()}}
async function solve(){document.getElementById('res').textContent='계산 중...';const r=await post('/solve');
 document.getElementById('res').textContent=JSON.stringify(r,null,1);refresh()}
async function load(){const r=await (await fetch('/state')).json();
 document.getElementById('pts').innerHTML=r.points.map(p=>`<button class="pt ${p.id==sel?'sel':''} ${p.clicked?'done':''}" onclick="sel='${p.id}';load()">${p.clicked?'✔':'·'} ${p.id} <span class=${p.check?'chk':''}>(${p.xyz.join(', ')})${p.check?' 검증':''}</span></button>`).join('');
 if(r.result&&!document.getElementById('res').textContent)document.getElementById('res').textContent=JSON.stringify(r.result,null,1)}
load();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("NAME", args.name)


if __name__ == "__main__":
    print(f"{args.name}: 입력 {args.src} {SIZE[0]}x{SIZE[1]}, 기준점 {len(WORLD)}개, 이미 클릭 {len(clicks)}개 "
          f"→ http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
