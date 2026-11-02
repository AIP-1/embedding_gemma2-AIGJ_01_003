"""10-3~10-5, 10-7 실제 배치 계산 + 카메라 배치 비교 (사람 대신 3D 관절 점을 실제 보정 파일로 투영)

- 카메라: 팀원 위치 보정(calib/cam0.npz = VIGI 848x480, cam1.npz = C920 640x480)을 '진짜 카메라'로 쓴다.
- 사람: 어깨 높이 1.43m(앉으면 1.05m), 발목 0.08m. 관절 화면 좌표에 2px 잡음(안내문 시뮬레이션과 같음).
- 계산: proximity_app과 같은 stereo.py 함수
    ankle1 : 발목 → 바닥(z=0) (팀원처럼 C920 1대)
    stereo : 두 카메라 어깨 중심 삼각측량
    fused 단독 : 한 카메라(C920) 어깨 → 높이 1.2m 평면 (H_GUESS)
- 책상: 사람 앞(카메라 쪽) 0.45m, 높이 0.74m. 발목이 가려지면 ankle1은 책상 다리(책상 앞쪽 바닥)를 발목으로 잡는다.
- 배치 비교: 같은 렌즈 값으로 카메라 위치만 바꿔 stereo 오차를 비교한다.
사용법: python sim_layout.py   → results/sim_layout.json
"""
import json

import numpy as np

import stereo as S

rng = np.random.default_rng(0)
TARGET = np.array([1.8, 0.0])
NOISE = 2.0          # 관절 오차(px)
N = 300              # 지점마다 반복
SH_STAND, SH_SIT, ANK = 1.43, 1.05, 0.08
cams = [S.load_cam(f"calib/cam{n}.npz") for n in (0, 1)]
SIZES = [(848, 480), (640, 480)]


def visible(cam, size, X):
    p = S.project_distorted(cam, X)
    Xc = cam["R"] @ np.asarray(X, float) + cam["t"].ravel()
    return Xc[2] > 0.2 and 0 <= p[0] < size[0] and 0 <= p[1] < size[1]


def obs(cam, X):
    """관절 화면 좌표(왜곡 있음) + 잡음 → 왜곡 보정한 좌표"""
    p = S.project_distorted(cam, X) + rng.normal(0, NOISE, 2)
    return S.undistort(cam, [p])[0]


def run_point(cs, sizes, xy, sit=False, desk=False):
    """한 지점에서 세 방식의 타겟까지 거리 분포 → 중앙값, 평균 오차"""
    xy = np.asarray(xy, float)
    sh = np.array([*xy, SH_SIT if sit else SH_STAND])
    ank = np.array([*xy, ANK])
    d_true = float(np.linalg.norm(xy - TARGET))
    toward = (cs[1]["C"][:2] - xy) / np.linalg.norm(cs[1]["C"][:2] - xy)     # C920 쪽 방향
    leg = np.array([*(xy + 0.45 * toward), 0.0])                             # 책상 앞다리 바닥점
    out = {}
    v = [visible(c, s, sh) for c, s in zip(cs, sizes)]
    res = {"ankle1": [], "stereo": [], "fused_single": []}
    for _ in range(N):
        a = leg if (sit or desk) else ank                                   # 가려지면 책상 다리를 발목으로
        if visible(cs[1], sizes[1], a):
            X = S.ray_plane(cs[1], obs(cs[1], a), 0.0)
            if X is not None:
                res["ankle1"].append(np.linalg.norm(X[:2] - TARGET))
        if all(v):
            X, err = S.triangulate(cs[0], cs[1], obs(cs[0], sh), obs(cs[1], sh))
            if err <= 15 and 0.5 <= X[2] <= 2.1:                             # match_people 조건과 같음
                res["stereo"].append(np.linalg.norm(X[:2] - TARGET))
        if v[1]:
            X = S.ray_plane(cs[1], obs(cs[1], sh), 1.2)
            if X is not None:
                res["fused_single"].append(np.linalg.norm(X[:2] - TARGET))
    for k, ds in res.items():
        out[k] = None if not ds else {"거리": round(float(np.median(ds)), 2),
                                      "평균오차": round(float(np.mean(np.abs(np.array(ds) - d_true))), 2),
                                      "보인비율": round(len(ds) / N, 2)}
    return d_true, out, v


# ---------------- 1) 실제 배치 10-7 표 ----------------
U = np.array([1.0, 0.0])                       # 타겟에서 카메라 반대쪽(앞쪽)으로
tests = [(f"{d}m, 서 있음", TARGET + d * U, False, False) for d in (0.5, 1.0, 2.0, 3.0)]
tests += [("1.0m, 책상에 앉음", TARGET + 1.0 * U, True, False), ("2.0m, 책상 뒤에 서 있음", TARGET + 2.0 * U, False, True)]
# 한 카메라(C920)에만 보이는 위치 찾기
only1 = None
for x in np.arange(1.0, 6.0, 0.1):
    for y in np.arange(-3.0, 3.0, 0.1):
        X = [x, y, SH_STAND]
        if visible(cams[1], SIZES[1], X) and not visible(cams[0], SIZES[0], X):
            if only1 is None or abs(np.linalg.norm([x - TARGET[0], y]) - 2.0) < abs(np.linalg.norm(only1 - TARGET) - 2.0):
                only1 = np.array([x, y])
if only1 is not None:
    tests.append((f"한 카메라(C920)에만 보이는 위치 ({np.linalg.norm(only1 - TARGET):.1f}m)", only1, False, False))
table = []
for name, xy, sit, desk in tests:
    d, out, v = run_point(cams, SIZES, xy, sit, desk)
    table.append({"name": name, "xy": np.round(xy, 2).tolist(), "d_true": round(d, 2), **out})
    print(name, json.dumps(out, ensure_ascii=False))

# 팀원 실측 자리 (카메라에서 약 2.6m, 단상 0.4m): stereo 오차 범위
C = (cams[0]["C"] + cams[1]["C"]) / 2
dirx = np.array([1.0, 0.0])
spot = C[:2] + 2.6 * dirx
sh = np.array([*spot, 0.4 + SH_STAND])
errs, hs = [], []
for _ in range(N):
    X, _ = S.triangulate(cams[0], cams[1], obs(cams[0], sh), obs(cams[1], sh))
    errs.append(np.linalg.norm(X[:2] - spot)); hs.append(X[2])
team_spot = {"카메라에서_m": 2.6, "어깨_높이_m": round(float(sh[2]), 2),
             "stereo_위치오차_중앙값_cm": round(100 * float(np.median(errs)), 1),
             "stereo_위치오차_90%_cm": round(100 * float(np.percentile(errs, 90)), 1),
             "stereo_높이_중앙값_m": round(float(np.median(hs)), 2), "높이_90%_범위_m": [round(float(np.percentile(hs, 5)), 2), round(float(np.percentile(hs, 95)), 2)]}
print("팀원 자리", team_spot)


# ---------------- 2) 카메라 배치 비교 (stereo, 서 있는 사람) ----------------
def look_at(C, target, size, K, dist):
    z = np.asarray(target, float) - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.vstack([x, y, z]); t = (-R @ C).reshape(3, 1)
    return {"K": K, "dist": dist, "R": R, "t": t, "P": K @ np.hstack([R, t]), "C": C, "Kinv": np.linalg.inv(K)}


Ks = [(cams[n]["K"], cams[n]["dist"]) for n in (0, 1)]
mid = TARGET + 1.5 * U                                         # 측정 지점들의 가운데를 바라봄
layouts = {
    "실제 (간격 0.38m, 높이 0.8~1.0m, 둘 다 앞)": cams,
    "간격 1.5m, 높이 1.0m, 가운데를 봄": [look_at(np.array([-0.8, 0.75, 1.0]), [*mid, 1.2], SIZES[0], *Ks[0]),
                                      look_at(np.array([-0.8, -0.75, 1.0]), [*mid, 1.2], SIZES[1], *Ks[1])],
    "간격 3m, 높이 2.4m, 가운데를 봄": [look_at(np.array([-0.8, 1.5, 2.4]), [*mid, 1.0], SIZES[0], *Ks[0]),
                                    look_at(np.array([-0.8, -1.5, 2.4]), [*mid, 1.0], SIZES[1], *Ks[1])],
    "안내문 권장: 옆 두 모서리(시선 약 90도), 높이 2.4m": [look_at(np.array([2.3, 3.0, 2.4]), [*mid, 1.0], SIZES[0], *Ks[0]),
                                                 look_at(np.array([2.3, -3.0, 2.4]), [*mid, 1.0], SIZES[1], *Ks[1])],
}
compare = {}
for lname, cs in layouts.items():
    row = {}
    for d in (0.5, 1.0, 2.0, 3.0):
        _, out, v = run_point(cs, SIZES, TARGET + d * U)
        row[f"{d}m"] = out["stereo"]["평균오차"] if out["stereo"] else None
    base = float(np.linalg.norm(cs[0]["C"] - cs[1]["C"]))
    compare[lname] = {"간격_m": round(base, 2), "stereo_평균오차_m": row}
    print(lname, compare[lname])

json.dump({"조건": {"관절오차_px": NOISE, "반복": N, "어깨_섬/앉음_m": [SH_STAND, SH_SIT], "타겟": TARGET.tolist(),
                  "ankle1·fused 카메라": "cam1 C920"},
           "실제배치_10-7": table, "팀원_실측_자리": team_spot, "배치_비교": compare},
          open("results/sim_layout.json", "w"), ensure_ascii=False, indent=1)
print("saved results/sim_layout.json")
