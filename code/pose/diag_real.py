import os, sys, glob, json, collections
os.environ["POSE_OFFLINE"] = "1"; os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); sys.argv = sys.argv[:1]
import numpy as np
from pose_model import body, facing_score, frame_features, load_model, predict, KP_CONF, LW, RW, LS, RS
LE, RE = 7, 8
out = {}
gru = {"v1": load_model("models/action.pt"), "v2": load_model("models_v2/action.pt")}
for p in sorted(glob.glob("real/kps_*.npz")):
    name = os.path.basename(p)[4:-4]
    d = np.load(p, allow_pickle=True)
    T, IDS, KPS, BOX = d["t"], list(d["ids"]), list(d["kps"]), list(d["boxes"])
    n = len(T); seen = 0; bodyok = 0; sw = []; wr_conf = []; rel = {"L": [], "R": []}; raised = collections.Counter()
    feats = []; prev = None; tprev = None
    for t, ids, kps, boxes in zip(T, IDS, KPS, BOX):
        if not len(ids):
            continue
        seen += 1
        big = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])); k = np.asarray(kps[big])
        wr_conf.append((k[LW, 2], k[RW, 2]))
        b = body(k)
        if b is None:
            continue
        bodyok += 1; (cx, cy), s = b; sw.append(s)
        for side, w in (("L", LW), ("R", RW)):
            if k[w, 2] > KP_CONF:
                rel[side].append((cy - k[w, 1]) / s)      # +면 손목이 어깨보다 위 (어깨 너비 단위)
        lu = k[LW, 2] > KP_CONF and k[LW, 1] < cy - 0.3 * s
        ru = k[RW, 2] > KP_CONF and k[RW, 1] < cy - 0.3 * s
        raised["둘다" if lu and ru else "하나" if lu or ru else "없음"] += 1
        spd = 0.0 if prev is None else float(np.linalg.norm(b[0]-prev)/s/max(t-tprev, 1e-3)); prev, tprev = b[0], t
        feats.append(frame_features(k, b, facing_score(k), spd))
    pct = lambda a, q: None if len(a) == 0 else round(float(np.percentile(a, q)), 2)
    g = {}
    for tag, m in gru.items():
        W = m["window"]; preds = collections.Counter(); probs = []
        for i in range(W, len(feats) + 1, 5):
            lab, pr = predict(m, feats[i-W:i]); preds[lab] += 1; probs.append(pr)
        g[tag] = {"예측": dict(preds), "확률_중앙값": pct(probs, 50), "0.7_넘은_%": None if not probs else round(100*np.mean(np.array(probs) >= 0.7), 1)}
    wc = np.array(wr_conf) if wr_conf else np.zeros((0, 2))
    out[name] = {"사람_보인_프레임": seen, "어깨_둘다_보임_%": round(100*bodyok/max(seen, 1), 1),
                 "어깨너비_px_중앙값": pct(sw, 50),
                 "손목_신뢰도_중앙값(L,R)": [pct(wc[:, 0], 50) if len(wc) else None, pct(wc[:, 1], 50) if len(wc) else None],
                 "손목_높이_어깨위_p50/p90(L)": [pct(rel["L"], 50), pct(rel["L"], 90)],
                 "손목_높이_어깨위_p50/p90(R)": [pct(rel["R"], 50), pct(rel["R"], 90)],
                 "규칙_올린팔_프레임": dict(raised), "GRU": g}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
json.dump(out, open("real/diagnose.json", "w"), ensure_ascii=False, indent=1)
