"""10-1 렌즈 보정 (카메라마다 1번): 체커보드를 여러 각도로 비추면 코너를 모아 calibrateCamera
사용법 (보드에서, 카메라 하나씩)
  python calib_intrinsics.py --src 0 --name cam0              # C920, 640x480
  python calib_intrinsics.py --src vigi:stream2 --name cam1   # VIGI 서브 848x480
  → 브라우저 http://127.0.0.1:8001  (Mac에서는 ssh -L 8001:127.0.0.1:8001 jetson)
진행
  - 체커보드를 화면 가운데·네 귀퉁이·가장자리로 옮기고, 앞뒤로 기울여 가며 비춘다.
  - 코너가 잡히고 0.5초 동안 흔들리지 않으면, 이전과 충분히 다른 자세일 때 자동으로 1장 모은다(사진이 아니라 코너 좌표만).
  - 20장쯤 모이면 [보정 계산] → RMS 0.5px 이하면 좋음, 1px 넘으면 다시.
결과: calib/<name>_intr.npz (K, dist, size, rms), calib/<name>_corners.npz (모은 코너)
"""
import argparse
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

import camio

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="0 (USB) 또는 vigi:stream2")
ap.add_argument("--name", required=True, help="cam0 / cam1 (앱의 --cams 순서와 같게)")
ap.add_argument("--res", default="640x480", help="USB 카메라 해상도 (VIGI는 무시)")
ap.add_argument("--focus", type=int, default=0, help="USB 카메라 고정 초점 값 (앱과 같게)")
ap.add_argument("--board", default="9x6", help="안쪽 코너 수 (가로x세로)")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변(m). 인쇄물을 재서 넣는다")
ap.add_argument("--target", type=int, default=20, help="모을 장수")
ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()
BW, BH = map(int, args.board.split("x"))
OBJ = np.zeros((BW * BH, 3), np.float32)
OBJ[:, :2] = np.mgrid[0:BW, 0:BH].T.reshape(-1, 2) * args.square

src = camio.Source(args.src, tuple(map(int, args.res.split("x"))), focus=args.focus)
first = camio.wait_frame(src)
SIZE = (first.shape[1], first.shape[0])
views = []                 # [(코너 (N,1,2) float32, 자세 특징)]
state = {"found": False, "msg": "체커보드를 비춰 주세요", "result": None, "auto": True}
jpeg = [None]
lock = threading.Lock()
cover = np.zeros((SIZE[1] // 20 + 1, SIZE[0] // 20 + 1), np.uint8)     # 코너가 지나간 칸 (20px 격자)
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def pose_feat(c):
    """자세 비교용: 중심(화면 비율), 크기, 가로·세로 기울기"""
    c = c.reshape(BH, BW, 2)
    ctr = c.reshape(-1, 2).mean(0) / SIZE
    w = np.linalg.norm(c[:, -1] - c[:, 0], axis=1)        # 행마다 가로 길이
    h = np.linalg.norm(c[-1] - c[0], axis=1)              # 열마다 세로 길이
    return np.array([ctr[0], ctr[1], np.sqrt(w.mean() * h.mean()) / SIZE[0],
                     (w[-1] - w[0]) / w.mean(), (h[-1] - h[0]) / h.mean()])


def different(f):
    if not views:
        return True
    d = min(np.abs(f - v[1]) @ np.array([1, 1, 1.5, 1, 1]) for v in views)
    return d > 0.12


def add_view(c):
    with lock:
        views.append((c.copy(), pose_feat(c)))
        for x, y in c.reshape(-1, 2):
            cover[int(y) // 20, int(x) // 20] = 1
    state["msg"] = f"{len(views)}장 모음"


def detect_loop():
    prev, still_since = None, None
    while True:
        t, f = src.latest()
        if f is None:
            time.sleep(0.05); continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(g, (BW, BH), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                                          + cv2.CALIB_CB_FAST_CHECK)
        vis = f.copy()
        ov = cv2.resize(cover * 255, SIZE, interpolation=cv2.INTER_NEAREST)
        vis[ov > 0] = (0.7 * vis[ov > 0] + 0.3 * np.array([0, 160, 0])).astype(np.uint8)
        state["found"] = bool(ok)
        if ok:
            c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), CRIT)
            moved = np.abs(c - prev).max() if prev is not None and prev.shape == c.shape else 99
            still_since = (still_since or time.time()) if moved < 1.5 else None
            if (state["auto"] and still_since and time.time() - still_since > 0.5
                    and len(views) < 60 and different(pose_feat(c))):
                add_view(c); still_since = None
            cv2.drawChessboardCorners(vis, (BW, BH), c, ok)
            prev = c
        else:
            prev, still_since = None, None
        cv2.putText(vis, f"{args.name} {SIZE[0]}x{SIZE[1]}  views {len(views)}/{args.target}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        jpeg[0] = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        state["last_corners"] = c if ok else None
        time.sleep(0.03)


def calibrate():
    with lock:
        imgp = [v[0] for v in views]
    if len(imgp) < 8:
        return {"error": f"최소 8장이 필요합니다 (지금 {len(imgp)}장)"}
    objp = [OBJ] * len(imgp)
    rms, K, dist, rv, tv = cv2.calibrateCamera(objp, imgp, SIZE, None, None)
    per = []
    for o, i, r, t in zip(objp, imgp, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(((p - i) ** 2).sum(-1).mean())))
    dropped = 0
    if len(imgp) >= 12:                       # 오차가 큰 장(흔들림·잘못 잡힘)은 빼고 한 번 더
        keep = [k for k, e in enumerate(per) if e < max(1.0, 2.5 * np.median(per))]
        dropped = len(imgp) - len(keep)
        if dropped:
            imgp = [imgp[k] for k in keep]
            rms, K, dist, rv, tv = cv2.calibrateCamera([OBJ] * len(imgp), imgp, SIZE, None, None)
    fovx = float(2 * np.degrees(np.arctan(SIZE[0] / (2 * K[0, 0]))))
    np.savez(f"calib/{args.name}_intr.npz", K=K, dist=dist, size=np.array(SIZE), rms=rms,
             src=args.src, focus=args.focus if src.kind == "usb" else -1, square=args.square)
    np.savez(f"calib/{args.name}_corners.npz", corners=np.array([v[0] for v in views]), size=np.array(SIZE),
             board=np.array([BW, BH]), square=args.square)
    res = {"rms": round(float(rms), 3), "views": len(imgp), "dropped": dropped,
           "fx": round(float(K[0, 0]), 1), "fy": round(float(K[1, 1]), 1),
           "cx": round(float(K[0, 2]), 1), "cy": round(float(K[1, 2]), 1),
           "fov_x_deg": round(fovx, 1), "dist": np.round(dist.ravel(), 4).tolist(),
           "coverage_%": round(100 * float(cover.mean()), 1),
           "saved": f"calib/{args.name}_intr.npz",
           "verdict": "좋음" if rms <= 0.5 else "쓸 만함" if rms <= 1.0 else "다시 하세요 (흔들림·인쇄 휨 확인)"}
    state["result"] = res
    print(res, flush=True)
    return res


app = FastAPI()


@app.get("/stream")
def stream():
    return StreamingResponse(camio.mjpeg(lambda: jpeg[0]), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def get_state():
    return {"views": len(views), "target": args.target, "found": state["found"], "msg": state["msg"],
            "auto": state["auto"], "result": state["result"], "coverage": round(100 * float(cover.mean()), 1)}


@app.post("/snap")
def snap():
    c = state.get("last_corners")
    if c is None:
        return {"ok": False, "msg": "코너가 안 잡힌 상태입니다"}
    add_view(c)
    return {"ok": True}


@app.post("/undo")
def undo():
    with lock:
        if views:
            views.pop()
            cover[:] = 0
            for v in views:
                for x, y in v[0].reshape(-1, 2):
                    cover[int(y) // 20, int(x) // 20] = 1
    state["msg"] = f"{len(views)}장"
    return {"ok": True}


@app.post("/auto")
def toggle_auto():
    state["auto"] = not state["auto"]
    return {"auto": state["auto"]}


@app.post("/calibrate")
def do_calibrate():
    return calibrate()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>렌즈 보정 NAME</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee;max-width:900px}img{width:100%;border-radius:6px}
button{font-size:15px;padding:6px 12px;margin:4px 4px 4px 0}.bar{height:10px;background:#333;border-radius:5px}
.bar>div{height:10px;background:#4c4;border-radius:5px}pre{background:#1b1b1b;padding:8px;white-space:pre-wrap}</style>
<h2>렌즈 보정 — NAME</h2>
<p>체커보드를 <b>가운데·네 귀퉁이·가장자리</b>로 옮기고 <b>앞뒤로 기울여</b> 비추세요. 잠깐 멈추면 자동으로 모읍니다. 초록 칸 = 코너가 지나간 곳 (화면 전체를 채울수록 좋음)</p>
<img src="/stream">
<p><span id=st></span></p><div class=bar><div id=bar style="width:0"></div></div>
<p><button onclick="post('/snap')">지금 1장 모으기</button><button onclick="post('/undo')">마지막 장 빼기</button>
<button id=au onclick="post('/auto')">자동 모으기</button><button onclick="calib()">보정 계산</button></p>
<pre id=res></pre>
<script>
async function post(u){return (await fetch(u,{method:'POST'})).json()}
async function calib(){document.getElementById('res').textContent='계산 중...';
 const r=await post('/calibrate');document.getElementById('res').textContent=JSON.stringify(r,null,1)}
async function poll(){const r=await (await fetch('/state')).json();
 document.getElementById('st').textContent=`${r.views}/${r.target}장, 화면 덮음 ${r.coverage}% — ${r.found?'코너 잡힘':'코너 안 잡힘'} — ${r.msg}`;
 document.getElementById('bar').style.width=Math.min(100,100*r.views/r.target)+'%';
 document.getElementById('au').textContent='자동 모으기: '+(r.auto?'켜짐':'꺼짐');}
setInterval(poll,500);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("NAME", args.name)


if __name__ == "__main__":
    print(f"{args.name}: 입력 {args.src} {SIZE[0]}x{SIZE[1]}, 초점 {src.focus} → http://127.0.0.1:{args.port}", flush=True)
    threading.Thread(target=detect_loop, daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
