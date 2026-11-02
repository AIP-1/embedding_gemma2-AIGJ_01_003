"""10-2 대안: 바닥에 놓은 체커보드 하나로 두 카메라 위치 보정 (줄자·클릭 없이)
체커보드 코너(칸 크기를 아는 점 54개)를 두 카메라가 동시에 찾아, 각 카메라의 위치·방향을 solvePnP로 구한다.
좌표계: 원점 = 보드의 첫 코너(화면에 빨간 원), x = 보드 가로(코너 9개 방향), y = 보드 세로, z = 위(+), 단위 m
사용법 (보드에서)
  python calib_board.py --cams 0 vigi:stream2 --square 0.024
  → 브라우저 http://127.0.0.1:8003
진행
  1) 체커보드를 두 카메라에 모두 보이는 바닥(사람이 다닐 자리 가운데쯤)에 평평하게 놓는다.
  2) 두 화면 모두 "코너 잡힘"이면 [기준 잡기] → calib/cam0.npz, cam1.npz, world_points.json 저장
  3) 보드를 다른 자리(1~2m 옮김, 책상 위 등)로 옮기고 [검증] → 두 카메라 삼각측량으로
     칸 간격(실제 square)·보드 평평함·높이 오차를 보여 준다. 여러 자리에서 반복하면 좋다.
렌즈 보정 파일: calib/cam0_intr.npz, calib/cam1_intr.npz (키트 형식 res 또는 size 둘 다 읽음)
프레임은 메모리에만 두고 디스크에 저장하지 않는다.
"""
import argparse
import json
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

import camio
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--cams", nargs=2, default=["0", "vigi:stream2"], help="cam0 cam1 입력 (앱과 같은 순서)")
ap.add_argument("--res", default="640x480", help="USB 카메라 해상도")
ap.add_argument("--focus", type=int, default=0)
ap.add_argument("--board", default="9x6", help="안쪽 코너 수 (가로x세로)")
ap.add_argument("--square", type=float, default=0.024, help="칸 한 변(m)")
ap.add_argument("--room-pad", type=float, default=1.0, help="방 범위 = 카메라·보드를 감싸는 사각형 + 이만큼(m)")
ap.add_argument("--port", type=int, default=8003)
ap.add_argument("--host", default="127.0.0.1")
args = ap.parse_args()
BW, BH = map(int, args.board.split("x"))
OBJ = np.zeros((BW * BH, 3), np.float64)
OBJ[:, :2] = np.mgrid[0:BW, 0:BH].T.reshape(-1, 2) * args.square
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def load_intr(n):
    d = np.load(f"calib/cam{n}_intr.npz")
    size = tuple(int(v) for v in (d["size"] if "size" in d else d["res"]))
    return d["K"].astype(np.float64), d["dist"].astype(np.float64), size


INTR = [load_intr(n) for n in range(2)]
SRCS = [camio.Source(s, tuple(map(int, args.res.split("x"))), focus=args.focus) for s in args.cams]
for n, s in enumerate(SRCS):
    f = camio.wait_frame(s)
    if (f.shape[1], f.shape[0]) != INTR[n][2]:
        raise SystemExit(f"cam{n}: 렌즈 보정 해상도 {INTR[n][2]} ≠ 입력 {(f.shape[1], f.shape[0])}")
det = [None, None]           # 카메라별 마지막 검출 코너 (N,2) 또는 None
jpegs = [None, None]
state = {"result": None, "checks": []}
lock = threading.Lock()


def detect(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(g, (BW, BH), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                                      + cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None
    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), CRIT).reshape(-1, 2)
    # 코너 순서를 두 카메라에서 같게: 첫 코너가 보드의 같은 모서리여야 한다.
    # findChessboardCorners는 180° 뒤집힌 순서로 줄 수 있고, 카메라가 반대쪽에서 보면 "화면 위쪽"도 다른 모서리다.
    # 10x7칸 보드는 양 끝 모서리 칸의 색이 다르다 → 첫 코너 바깥 칸이 '검정'이 되도록 순서를 통일
    if corner_square_gray(g, c, 0, BW + 1) > corner_square_gray(g, c, len(c) - 1, len(c) - BW - 2):
        c = c[::-1].copy()
    return c


def corner_square_gray(g, c, i, j):
    """코너 i에서 대각선 안쪽 코너 j의 반대 방향으로 반 칸 나간 곳(모서리 칸 가운데)의 밝기"""
    p = c[i] + 0.5 * (c[i] - c[j])
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = g.shape
    x0, x1, y0, y1 = max(x - 2, 0), min(x + 3, w), max(y - 2, 0), min(y + 3, h)
    return float(g[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 128.0


def loop(n):
    while True:
        t, f = SRCS[n].latest()
        if f is None:
            time.sleep(0.05); continue
        c = detect(f)
        det[n] = c
        vis = f.copy()
        if c is not None:
            cv2.drawChessboardCorners(vis, (BW, BH), c.reshape(-1, 1, 2).astype(np.float32), True)
            cv2.circle(vis, tuple(int(v) for v in c[0]), 9, (0, 0, 255), 2)       # 원점 코너
        res = state["result"]
        if res is not None:                                                     # 계산된 좌표축을 그려 확인
            cam = res["cams"][n]
            o = S.project_distorted(cam, [0, 0, 0])
            for ax, col in (([0.2, 0, 0], (0, 0, 255)), ([0, 0.2, 0], (0, 255, 0)), ([0, 0, 0.2], (255, 0, 0))):
                p = S.project_distorted(cam, ax)
                cv2.arrowedLine(vis, tuple(int(v) for v in o), tuple(int(v) for v in p), col, 2)
        cv2.putText(vis, f"cam{n} {'corners OK' if c is not None else 'no board'}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        jpegs[n] = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        time.sleep(0.05)


def averaged_corners(n, k=8):
    """흔들림을 줄이려고 연속 k번 검출의 평균"""
    acc = []
    end = time.time() + 4
    while len(acc) < k and time.time() < end:
        c = det[n]
        if c is not None:
            acc.append(c.copy())
        time.sleep(0.08)
    return np.mean(acc, axis=0) if len(acc) >= k // 2 else None


def pose(n, img):
    K, dist, _ = INTR[n]
    ok, rv, tv = cv2.solvePnP(OBJ, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rv, tv = cv2.solvePnPRefineLM(OBJ, img, K, dist, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    p, _ = cv2.projectPoints(OBJ, rv, tv, K, dist)
    err = float(np.linalg.norm(p.reshape(-1, 2) - img, axis=1).mean())
    return R, tv.reshape(3, 1), err


def make_cam(K, dist, R, t):
    return {"K": K, "dist": dist, "R": R, "t": t, "P": K @ np.hstack([R, t]), "C": (-R.T @ t).ravel(),
            "Kinv": np.linalg.inv(K)}


def set_reference():
    imgs = [averaged_corners(n) for n in range(2)]
    if any(i is None for i in imgs):
        return {"error": "두 카메라 모두에서 코너가 안정적으로 잡혀야 합니다: " +
                         ", ".join(f"cam{n} {'OK' if i is not None else '안 잡힘'}" for n, i in enumerate(imgs))}
    poses = [pose(n, imgs[n]) for n in range(2)]
    # 보드 좌표계의 z(= x축 × y축)가 바닥 아래를 향하면(카메라 z가 음수) 뒤집어 z를 위로
    C0 = (-poses[0][0].T @ poses[0][1]).ravel()
    flip = np.diag([1.0, -1.0, -1.0]) if C0[2] < 0 else np.eye(3)
    cams = []
    for n, (R, t, err) in enumerate(poses):
        Rw = R @ flip                                  # 세계(뒤집은 보드) → 카메라
        K, dist, size = INTR[n]
        cams.append(make_cam(K, dist, Rw, t))
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=Rw, t=t, size=np.array(size), res=np.array(size),
                 src=args.cams[n], method="checkerboard")
    # 원점 코너의 화면 위치가 두 카메라에서 같은 모서리인지 확인: 삼각측량한 보드 코너가 평면·간격과 맞아야 한다
    tri = triangulate_all(cams, imgs)
    corners_w = (flip @ OBJ.T).T
    err3d = np.linalg.norm(tri - corners_w, axis=1)
    cpos = [c["C"] for c in cams]
    xy = np.array([p[:2] for p in cpos] + [corners_w[0, :2], corners_w[-1, :2]])
    room = [float(xy[:, 0].min() - args.room_pad), float(xy[:, 1].min() - args.room_pad),
            float(xy[:, 0].max() + args.room_pad), float(xy[:, 1].max() + args.room_pad)]
    pts = [{"id": f"B{i}", "xyz": [round(float(v), 4) for v in corners_w[i]]}
           for i in (0, BW - 1, BW * (BH - 1), BW * BH - 1)]
    json.dump({"_설명": "체커보드 기준 좌표계 (calib_board.py). 원점=보드 첫 코너, x=보드 가로, y=보드 세로, z=위. "
                       "room = 앱이 사람을 찾는 바닥 범위 [x0, y0, x1, y1]",
               "room": room, "points": pts}, open("calib/world_points.json", "w"), ensure_ascii=False, indent=1)
    base = float(np.linalg.norm(cpos[0] - cpos[1]))
    ang = float(np.degrees(np.arccos(np.clip(
        np.dot(cpos[0] - corners_w.mean(0), cpos[1] - corners_w.mean(0)) /
        (np.linalg.norm(cpos[0] - corners_w.mean(0)) * np.linalg.norm(cpos[1] - corners_w.mean(0))), -1, 1))))
    res = {"저장": ["calib/cam0.npz", "calib/cam1.npz", "calib/world_points.json"],
           "카메라": [{"이름": f"cam{n} ({args.cams[n]})", "위치_m": np.round(cpos[n], 3).tolist(),
                     "높이_m": round(float(cpos[n][2]), 3), "보드까지_m": round(float(np.linalg.norm(cpos[n])), 2),
                     "재투영_평균_px": round(poses[n][2], 3)} for n in range(2)],
           "두_카메라_간격_m": round(base, 3), "보드에서_본_두_시선_각도_deg": round(ang, 1),
           "보드_코너_삼각측량_오차_mm": {"평균": round(1000 * float(err3d.mean()), 1),
                                    "최대": round(1000 * float(err3d.max()), 1)},
           "방_범위": [round(v, 2) for v in room],
           "판정": "좋음" if err3d.mean() < 0.005 and max(p[2] for p in poses) < 1.0 else
                   "점검 필요 (두 카메라 코너 순서가 다르거나 렌즈 보정이 안 맞음)"}
    with lock:
        state["result"] = {"cams": cams, "flip": flip, "summary": res}
        state["checks"] = []
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def triangulate_all(cams, imgs):
    und = [S.undistort(cams[n], imgs[n]) for n in range(2)]
    return np.array([S.triangulate(cams[0], cams[1], und[0][i], und[1][i])[0] for i in range(len(OBJ))])


def check():
    if state["result"] is None:
        return {"error": "먼저 [기준 잡기]를 하세요"}
    imgs = [averaged_corners(n) for n in range(2)]
    if any(i is None for i in imgs):
        return {"error": "두 카메라 모두에서 코너가 잡혀야 합니다"}
    cams = state["result"]["cams"]
    tri = triangulate_all(cams, imgs).reshape(BH, BW, 3)
    dx = np.linalg.norm(np.diff(tri, axis=1), axis=2).ravel()       # 가로 이웃 코너 간격
    dy = np.linalg.norm(np.diff(tri, axis=0), axis=2).ravel()
    sp = np.concatenate([dx, dy])
    diag = float(np.linalg.norm(tri[-1, -1] - tri[0, 0]))
    diag_true = float(np.hypot((BW - 1), (BH - 1)) * args.square)
    flat = tri.reshape(-1, 3)
    ctr = flat.mean(0)
    _, sv, vt = np.linalg.svd(flat - ctr)
    plane_rms = float(np.sqrt((sv[-1] ** 2) / len(flat)))
    tilt = float(np.degrees(np.arccos(abs(vt[-1][2]))))
    # 한 카메라만으로(바닥 높이 = 보드 높이 가정) 같은 보드 중심을 구했을 때와의 차이
    single = []
    for n in range(2):
        u = S.undistort(cams[n], [imgs[n].mean(0)])[0]
        X = S.ray_plane(cams[n], u, float(ctr[2]))
        single.append(None if X is None else round(100 * float(np.linalg.norm(X[:2] - ctr[:2])), 1))
    res = {"보드_중심_m": np.round(ctr, 3).tolist(), "원점에서_거리_m": round(float(np.linalg.norm(ctr[:2])), 2),
           "칸_간격_mm": {"실제": round(1000 * args.square, 1), "평균": round(1000 * float(sp.mean()), 2),
                       "표준편차": round(1000 * float(sp.std()), 2)},
           "대각선_오차_%": round(100 * (diag - diag_true) / diag_true, 2),
           "보드_평면_벗어남_mm": round(1000 * plane_rms, 2), "보드_기울기_deg": round(tilt, 1),
           "보드_높이_m": round(float(ctr[2]), 3),
           "한_카메라_추정과_차이_cm(cam0,cam1)": single}
    with lock:
        state["checks"].append(res)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


app = FastAPI()


@app.get("/stream/{n}")
def stream(n: int):
    return StreamingResponse(camio.mjpeg(lambda: jpegs[n]), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def get_state():
    r = state["result"]
    return {"found": [d is not None for d in det], "result": r["summary"] if r else None, "checks": state["checks"]}


@app.post("/reference")
def do_reference():
    return set_reference()


@app.post("/check")
def do_check():
    return check()


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>체커보드 위치 보정</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:10px;flex-wrap:wrap}
.row>div{flex:1;min-width:300px}img{width:100%;border-radius:6px}button{font-size:15px;padding:6px 12px;margin:3px 3px 3px 0}
pre{background:#1b1b1b;padding:8px;white-space:pre-wrap;font-size:12px}.ok{color:#5d5}.no{color:#f66}</style>
<h2>체커보드 위치 보정 (10-2)</h2>
<p>① 체커보드를 <b>두 카메라에 모두 보이는 바닥</b>에 평평하게 놓기 → 두 화면 모두 <span class=ok>코너 잡힘</span>이면
<button onclick="go('/reference')">기준 잡기</button><br>
② 보드를 <b>다른 자리</b>(1~2m 옮김, 책상 위 등)로 옮기고 <button onclick="go('/check')">검증</button> — 여러 번 반복 가능<br>
<small>빨간 원 = 원점 코너. 기준을 잡으면 좌표축이 그려집니다 (빨강 x, 초록 y, 파랑 z=위). 보드를 움직이는 동안에는 손이 코너를 가리지 않게.</small></p>
<p id=st></p>
<div class=row><div><img src="/stream/0"></div><div><img src="/stream/1"></div></div>
<h3>결과</h3><pre id=res>-</pre><h3>검증 기록</h3><pre id=chk>-</pre>
<script>
async function go(u){document.getElementById('res').textContent='계산 중...';
 const r=await (await fetch(u,{method:'POST'})).json();document.getElementById('res').textContent=JSON.stringify(r,null,1);poll()}
async function poll(){const r=await (await fetch('/state')).json();
 document.getElementById('st').innerHTML=r.found.map((f,i)=>`cam${i}: <span class=${f?'ok':'no'}>${f?'코너 잡힘':'안 잡힘'}</span>`).join(' &nbsp; ');
 if(r.checks.length)document.getElementById('chk').textContent=JSON.stringify(r.checks,null,1);}
setInterval(poll,700);poll();
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    for n in range(2):
        threading.Thread(target=loop, args=(n,), daemon=True).start()
    print(f"cam0={args.cams[0]} cam1={args.cams[1]}, 보드 {BW}x{BH} 칸 {args.square * 1000:.0f}mm → http://127.0.0.1:{args.port}",
          flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
