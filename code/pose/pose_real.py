"""9단계 실제 영상 검증: 사람 동작 영상에서 관절 좌표만 뽑아(영상·사진은 저장하지 않음) 9단계 판정 코드에 넣는다.

  python pose_real.py extract <영상 폴더>   # YOLO-pose + ByteTrack → real/kps_<이름>.npz (관절 좌표·추적 ID·박스만)
  python pose_real.py analyze              # real/kps_*.npz → 규칙 raw / 안정화 / GRU(v1, v2) 비교 → real/results.json

- 영상은 가로 1280으로 줄이고 약 15fps로 솎아 앱과 같은 조건으로 만든다.
- 거꾸로 단 카메라 영상은 첫 프레임들에서 코가 엉덩이보다 아래면 180도 돌린다.
- 정답은 파일 이름으로 정한다: arms_up → 만세, one_arm → 한팔들기, wave → 손흔들기, handshake → 악수.
  반복 횟수 정답은 없으므로 '그 동작으로 판정된 프레임 비율', '라벨이 바뀐 횟수(깜빡임)', '이벤트 수'를 비교한다.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

CMD = sys.argv[1] if len(sys.argv) > 1 else "analyze"
OUT = "real"
os.makedirs(OUT, exist_ok=True)
FPS = 15.0
EXPECT = {"arms_up": "만세", "one_arm": "한팔들기", "wave": "손흔들기", "handshake": "악수"}
GRU_MAP = {"arms_up": "arms_up", "wave": "wave"}          # GRU 라벨로 비교할 수 있는 것


def expected(name):
    for k, v in EXPECT.items():
        if name.startswith(k):
            return v
    return None


# ---------------------------------------------------------------- 1. 관절 추출
def extract(folder):
    import av
    import cv2
    from ultralytics import YOLO
    det = YOLO("../prox/yolo11n-pose.pt")
    files = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    summary = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        c = av.open(path)
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate or 25)
        step = max(1, round(fps / FPS))
        frames = []
        for i, fr in enumerate(c.decode(s)):
            if i % step:
                continue
            img = fr.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            img = cv2.resize(img, (1280, round(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            frames.append((i / fps, img))
        c.close()
        # 거꾸로 달린 카메라인지: 앞 프레임들에서 코(0)가 엉덩이(11,12)보다 아래인 사람이 많으면 180도
        votes = 0
        for _, img in frames[:20]:
            r = det.predict(img, conf=0.5, verbose=False)[0]
            if r.keypoints is None:
                continue
            for k in r.keypoints.data.cpu().numpy():
                if k[0, 2] > 0.5 and k[11, 2] > 0.5 and k[12, 2] > 0.5:
                    votes += 1 if k[0, 1] > (k[11, 1] + k[12, 1]) / 2 else -1
        flip = votes > 0
        det.predictor = None                                  # 추적기 초기화
        T, IDS, KPS, BOX = [], [], [], []
        for t, img in frames:
            if flip:
                img = cv2.rotate(img, cv2.ROTATE_180)
            r = det.track(img, persist=True, conf=0.5, verbose=False, tracker="bytetrack.yaml")[0]
            if r.boxes is None or r.boxes.id is None or r.keypoints is None:
                ids, kps, box = np.zeros(0, int), np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), int)
            else:
                ids = r.boxes.id.int().cpu().numpy()
                kps = r.keypoints.data.cpu().numpy().astype(np.float32)
                box = r.boxes.xyxy.int().cpu().numpy()
            T.append(t); IDS.append(ids); KPS.append(kps); BOX.append(box)
        np.savez_compressed(f"{OUT}/kps_{name}.npz", t=np.array(T), ids=np.array(IDS, dtype=object),
                            kps=np.array(KPS, dtype=object), boxes=np.array(BOX, dtype=object),
                            flip=flip, src_fps=fps, size=np.array([1280, frames[0][1].shape[0]]))
        n_people = [len(i) for i in IDS]
        summary[name] = {"프레임": len(T), "길이_s": round(T[-1], 1) if T else 0, "180도_회전": bool(flip),
                         "사람_수_중앙값": int(np.median(n_people)) if n_people else 0,
                         "사람_없는_프레임_%": round(100 * float(np.mean([n == 0 for n in n_people])), 1),
                         "추적_ID_수": len(set(int(x) for ids in IDS for x in ids))}
        print(name, summary[name], flush=True)
    json.dump(summary, open(f"{OUT}/extract.json", "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 2. 판정
def load(name):
    d = np.load(f"{OUT}/kps_{name}.npz", allow_pickle=True)
    return d["t"], list(d["ids"]), list(d["kps"]), list(d["boxes"])


def run_rules(pa, name, mode, models=(), behavior=False):
    """pose_app.step()을 그대로 호출. 프레임마다 '가장 큰 사람'의 라벨을 기록"""
    for d in (pa.tracks, pa.pairs, pa.counts, pa.beh_counts):
        d.clear()
    pa.pair_on.clear(); pa.events.clear()
    pa.MODE = mode; pa.args.behavior = behavior; pa.models = list(models)
    pa.SEQ_LEN = max([m["window"] for m in models], default=1)
    T, IDS, KPS, BOX = load(name)
    T0 = 1_000_000.0
    labels, pair_frames = [], 0
    for t, ids, kps, boxes in zip(T, IDS, KPS, BOX):
        ids = [int(i) for i in ids]
        pa.step(ids, np.asarray(kps) if len(ids) else [], [tuple(int(v) for v in b) for b in boxes], T0 + t)
        if len(ids):
            big = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            st = pa.tracks.get(ids[big])
            labels.append(st["label"] if st else None)
        else:
            labels.append(None)
        pair_frames += bool(pa.pair_on)
    pa.step([], [], [], T0 + T[-1] + 5)                       # 추적 종료 처리
    ev = collections.Counter(e["label"] for e in pa.events)
    return labels, dict(ev), pair_frames, len(T)


def flicker(labels):
    seq = [l for l in labels if l is not None]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def analyze():
    os.environ["POSE_OFFLINE"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.argv = sys.argv[:1]
    import pose_app as pa
    from pose_model import load_model
    gru = {"v1": load_model("models/action.pt"), "v2": load_model("models_v2/action.pt")}
    names = sorted(os.path.basename(p)[4:-4] for p in glob.glob(f"{OUT}/kps_*.npz"))
    res = {}
    for name in names:
        exp = expected(name)
        r = {"정답": exp}
        for mode in ("raw", "smooth"):
            labels, ev, pair_frames, n = run_rules(pa, name, mode)
            seen = [l for l in labels if l is not None]
            row = {"이벤트": ev, "라벨_바뀜": flicker(labels), "사람_보인_프레임": len(seen),
                   "라벨_분포_%": {k: round(100 * v / max(len(seen), 1), 1) for k, v in collections.Counter(seen).most_common()}}
            if exp == "악수":
                row["악수_프레임_%"] = round(100 * pair_frames / n, 1)
            elif exp:
                row["정답_라벨_프레임_%"] = round(100 * sum(1 for l in seen if l == exp) / max(len(seen), 1), 1)
            r[mode] = row
        for tag, m in gru.items():
            labels, ev, _, _ = run_rules(pa, name, "smooth", models=[m])
            seen = [l for l in labels if l is not None]
            row = {"이벤트": ev, "라벨_분포_%": {k: round(100 * v / max(len(seen), 1), 1)
                                              for k, v in collections.Counter(seen).most_common()},
                   "학습_라벨": m["labels"]}
            r[f"GRU_{tag}"] = row
        res[name] = r
        print(name, json.dumps({k: (v if k == "정답" else {kk: vv for kk, vv in v.items() if kk != "라벨_분포_%"})
                                for k, v in r.items()}, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if CMD == "extract":
        extract(sys.argv[2])
    else:
        analyze()
