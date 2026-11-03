import numpy as np, cv2, stereo as S, calib_board as B
cs = [np.asarray(c, np.float64) for c in np.load("/tmp/t1_corners.npy", allow_pickle=True)]
CAL = "/home/team2-1/eg2-lab/prox/calib/"
obj = B.board_points(9, 6, 0.024)
v = np.load(CAL + "cam0_intr.npz"); c_fix = np.load(CAL + "cam1_intr.npz"); c_free = np.load(CAL + "cam1_intr_free.npz")
saved = [S.load_cam(CAL + f"cam{n}.npz") for n in (0, 1)]
print("저장된 위치 C0", saved[0]["C"].round(3), "C1", saved[1]["C"].round(3),
      "간격", round(float(np.linalg.norm(saved[0]["C"] - saved[1]["C"])), 3))

def mk(R, t, K, D):
    return {"K": K, "dist": D, "R": R, "t": t, "P": K @ np.hstack([R, t]), "C": (-R.T @ t).ravel(), "Kinv": np.linalg.inv(K)}

def run(name, K1, D1, z):
    imgs = [cs[0].copy(), cs[1].copy()]
    out = B.solve(obj, imgs, [v["K"], K1], [v["dist"], D1], False, z)
    cams = [mk(R, t, K, D) for (R, t, e), K, D in zip(out, [v["K"], K1], [v["dist"], D1])]
    und = [S.undistort(cams[n], imgs[n]) for n in (0, 1)]
    tr = [S.triangulate(cams[0], cams[1], und[0][i], und[1][i]) for i in range(54)]
    X = np.array([a for a, _ in tr]); rerr = np.array([b for _, b in tr])
    g = X.reshape(6, 9, 3)
    sp = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    base = np.linalg.norm(cams[0]["C"] - cams[1]["C"])
    print(f"[{name}] C0 {cams[0]['C'].round(3)} C1 {cams[1]['C'].round(3)} 간격 {base:.3f}m | PnP 재투영 {out[0][2]:.2f}/{out[1][2]:.2f}px"
          f" | 삼각측량 재투영 {rerr.mean():.2f}px | 칸 {1000*sp.mean():.2f}±{1000*sp.std():.2f}mm (실제 24) | 보드 중심 {X.mean(0).round(3)}")
    return cams

A = run("고정 값 (C920 고정 680.6)", c_fix["K"], c_fix["dist"], 1.0)
Bc = run("C920 자유 값 (669.9)", c_free["K"], c_free["dist"], 1.0)
# 같은 화면 좌표를 두 보정으로 삼각측량했을 때 사람 위치가 얼마나 달라지나: A로 투영한 가상 어깨점 → B로 역산
print("사람(어깨 1.4m) 위치 차이 — 같은 화면 좌표를 두 보정으로 계산했을 때")
for x in (1.0, 1.8, 2.8, 3.8, 4.8):
    P = np.array([x, 0.0, 1.4])
    p = [S.project(A[n], P) for n in (0, 1)]
    XB, e = S.triangulate(Bc[0], Bc[1], p[0], p[1])
    d_tgt_A = np.linalg.norm(P[:2] - [1.8, 0]); d_tgt_B = np.linalg.norm(XB[:2] - [1.8, 0])
    print(f"  x={x}m: 차이 {100*np.linalg.norm(XB[:2]-P[:2]):.1f}cm, 타겟(1.8,0)까지 거리 {d_tgt_A:.2f} → {d_tgt_B:.2f}m")
# 삼각측량 정밀도: 화면 좌표 1px 오차가 만드는 위치 오차 (자유 값 기준)
print("화면 1px 오차가 만드는 위치 오차 (어깨 높이 1.4m, y=0)")
rng = np.random.default_rng(0)
for x in (1.0, 1.8, 2.8, 3.8, 4.8):
    P = np.array([x, 0.0, 1.4]); p = [S.project(Bc[n], P) for n in (0, 1)]
    errs = []
    for _ in range(300):
        q = [p[n] + rng.normal(0, 1.0, 2) for n in (0, 1)]
        X, _ = S.triangulate(Bc[0], Bc[1], q[0], q[1]); errs.append(np.linalg.norm(X[:2] - P[:2]))
    print(f"  x={x}m (카메라에서 약 {x+0.8:.1f}m): 바닥 위치 오차 중앙값 {100*np.median(errs):.1f}cm, 90% {100*np.percentile(errs,90):.1f}cm")
np.savez(CAL + "cam1_free_pose.npz", R=Bc[1]["R"], t=Bc[1]["t"])
for n in (0, 1):
    np.savez(CAL + f"cam{n}_free.npz", K=Bc[n]["K"], dist=Bc[n]["dist"], R=Bc[n]["R"], t=Bc[n]["t"],
             res=np.array([848, 480] if n == 0 else [640, 480]))
print("저장: calib/cam0_free.npz, cam1_free.npz (C920 자유 값으로 다시 계산한 위치 보정)")
