"""10-6 타겟 자동 찾기 실측: 위치 보정 때 두 카메라가 동시에 찍은 체커보드 사진으로
YOLOE(문구) 후보 → EmbeddingGemma 2 재확인 → 두 카메라 박스 중심 삼각측량 → 정답(보드 중심 (0, 0, 1.0))과 비교.
proximity_app.find_target()과 같은 계산(conf 0.1, 16px 미만 제외, 점수 0.15 이상)이다.
사진은 메모리에서만 쓰고, 크롭은 임시 파일로 임베딩한 뒤 바로 지운다.
사용법: python target_board_test.py <board0.jpg> <board1.jpg>  → results/target_board.json
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import torch

import stereo as S

PROMPTS = ["checkerboard", "black and white checkerboard", "chessboard", "calibration pattern"]
TRUE = np.array([0.0, 0.0, 1.0])                      # 위치 보정 정의상 보드 중심 = 원점 위 1.0m
cams = [S.load_cam(f"calib/cam{n}.npz") for n in (0, 1)]
imgs = [cv2.imread(p) for p in sys.argv[1:3]]

# 정답 화면 위치: 체커보드 코너 54개의 평균
gt_px = []
for im in imgs:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ok, c = cv2.findChessboardCorners(g, (9, 6))
    gt_px.append(c.reshape(-1, 2).mean(0) if ok else None)

from ultralytics import YOLOE
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                          model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
yoloe = YOLOE("yoloe-11s-seg.pt")
tmp = tempfile.mkdtemp()
out = {}
for prompt in PROMPTS:
    yoloe.set_classes([prompt], yoloe.get_text_pe([prompt]))
    qv = np.asarray(emb.encode(f"a photo of a {prompt}", prompt_name="SearchQuery", normalize_embeddings=True), np.float32)
    found = []
    for n, f in enumerate(imgs):
        r = yoloe.predict(f, conf=0.1, verbose=False)[0]
        best, n_cand = None, 0
        for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.int().tolist()):
            if min(x2 - x1, y2 - y1) < 16:
                continue
            n_cand += 1
            p = os.path.join(tmp, f"c{n}_{i}.jpg")
            cv2.imwrite(p, f[y1:y2, x1:x2])
            v = np.asarray(emb.encode({"image": p}, normalize_embeddings=True), np.float32)
            os.remove(p)
            s = float(v @ qv)
            if best is None or s > best[0]:
                best = (s, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2))
        found.append((best, n_cand))
    row = {"후보_수": [c for _, c in found],
           "점수": [None if b is None else round(b[0], 3) for b, _ in found],
           "박스중심_정답과_px": [None if (b is None or g is None) else round(float(np.hypot(b[1][0] - g[0], b[1][1] - g[1])), 1)
                           for (b, _), g in zip(found, gt_px)]}
    ok = [b is not None and b[0] >= 0.15 for b, _ in found]
    if all(ok):
        p0 = S.undistort(cams[0], [found[0][0][1]])[0]; p1 = S.undistort(cams[1], [found[1][0][1]])[0]
        X, err = S.triangulate(cams[0], cams[1], p0, p1)
        row.update(방식="삼각측량", 좌표=np.round(X, 3).tolist(), 재투영_px=round(err, 2),
                   오차_cm=round(100 * float(np.linalg.norm(X - TRUE)), 1),
                   바닥오차_cm=round(100 * float(np.linalg.norm(X[:2] - TRUE[:2])), 1))
    elif any(ok):
        n = ok.index(True)
        X = S.ray_plane(cams[n], S.undistort(cams[n], [found[n][0][1]])[0], 0.8)     # 앱 기본 --target-h 0.8
        row.update(방식=f"cam{n} 단독 (높이 0.8m 가정)", 좌표=None if X is None else np.round(X, 3).tolist(),
                   바닥오차_cm=None if X is None else round(100 * float(np.linalg.norm(X[:2] - TRUE[:2])), 1))
    else:
        row["방식"] = "찾지 못함"
    out[prompt] = row
    print(prompt, json.dumps(row, ensure_ascii=False), flush=True)
os.rmdir(tmp)
# 정답 화면 위치를 삼각측량하면 (보정 자체의 오차)
if all(g is not None for g in gt_px):
    X, err = S.triangulate(cams[0], cams[1], S.undistort(cams[0], [gt_px[0]])[0], S.undistort(cams[1], [gt_px[1]])[0])
    out["_코너평균_삼각측량"] = {"좌표": np.round(X, 3).tolist(), "오차_cm": round(100 * float(np.linalg.norm(X - TRUE)), 1)}
    print("코너 평균", out["_코너평균_삼각측량"])
json.dump(out, open("results/target_board.json", "w"), ensure_ascii=False, indent=1)
