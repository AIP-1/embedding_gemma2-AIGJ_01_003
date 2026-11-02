"""설계 변수 바꾸기 실험: ① 어깨 한쪽만 보여도 판정 ② 팔 올림 기준 0.3 → 0.0 (어깨너비 단위)"""
import os, sys, json, collections
os.environ["POSE_OFFLINE"] = "1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); sys.argv = sys.argv[:1]
import numpy as np
import pose_app as pa, pose_model as pm
from pose_real import run_rules, flicker, expected, load
orig_body, orig_label = pm.body, pa.pose_label
LS, RS, NOSE = 5, 6, 0

def body_one(k):
    """어깨가 하나만 보이면: 중심 = 보이는 어깨, 너비 = 코-어깨 거리 x 1.6 (없으면 None)"""
    b = orig_body(k)
    if b is not None:
        return b
    vis = [i for i in (LS, RS) if k[i, 2] >= pm.KP_CONF]
    if len(vis) != 1 or k[NOSE, 2] < pm.KP_CONF:
        return None
    s = k[vis[0], :2]
    return s.copy(), max(1.6 * float(np.linalg.norm(k[NOSE, :2] - s)), 20.0)

def make_label(th):
    def lab(k, b):
        (cx, cy), sw = b
        up = lambda w: k[w, 2] > pm.KP_CONF and k[w, 1] < cy - th * sw
        lu, ru = up(pa.LW), up(pa.RW)
        return "만세" if lu and ru else "한팔들기" if lu or ru else "기본"
    return lab

res = {}
for variant, (bf, th) in {"원래(양쪽 어깨, 0.3)": (orig_body, 0.3), "① 한쪽 어깨 허용": (body_one, 0.3),
                          "② 올림 기준 0.0": (orig_body, 0.0), "①+②": (body_one, 0.0)}.items():
    pa.body = bf; pa.pose_label = make_label(th)
    for name in ("arms_up", "arms_up2", "one_arm", "wave", "wave2", "handshake", "pose_test"):
        exp = expected(name)
        labels, ev, pair_frames, n = run_rules(pa, name, "smooth")
        seen = [l for l in labels if l is not None]
        judged = sum(1 for l in seen)
        hit = (round(100 * pair_frames / n, 1) if exp == "악수" else
               round(100 * sum(1 for l in seen if l == exp) / max(len(seen), 1), 1) if exp else None)
        res.setdefault(name, {})[variant] = {"정답_프레임_%": hit, "이벤트": ev, "라벨_바뀜": flicker(labels)}
for name, r in res.items():
    print(name, json.dumps(r, ensure_ascii=False))
json.dump(res, open("real/fix_variants.json", "w"), ensure_ascii=False, indent=1)
