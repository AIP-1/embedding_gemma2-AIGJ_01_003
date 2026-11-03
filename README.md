# Jetson 멀티모달 임베딩 실습 (EmbeddingGemma 2)

Jetson Orin Nano Super 보드 한 대에서 `google/embeddinggemma-2`를 돌려 본 실습 기록입니다(team2).
- **검색**: 텍스트·이미지·동영상·음성
- **미세 조정**: 강의실 도메인 데이터로 추가 학습
- **실시간 검색**: 카메라 영상에서 물체를 검출하고, 웹에서 글자로 검색

👉 **자세한 결과는 [결과 보고서](docs/결과보고서.md)에 있습니다.**

## 핵심 결과

| 단계 | 결과 |
|---|---|
| 1 환경 준비 | NVMe로 부팅된 보드라 SSD 마운트는 생략. MAXN_SUPER·스왑 8GB, torch 2.14.1+cu132 GPU 행렬곱 확인, `torch-pin.txt`로 버전 고정 |
| 2 패키지·모델 | sentence-transformers 6.2.0, `eg2.py`의 `load()`로 필요한 인코더만 로드. torchaudio·torchcodec은 보드와 맞지 않아 제거 |
| 3 텍스트 검색 | 768·256·128차원 모두 2/2 정답, 질의 2개 165ms |
| 4 영상 검색 | 카메라 장면 사진 1/3 정답(△1), 이미지 5장 2.4초 |
| 5 음성 검색 | 2/3 정답(합성 음성만 오답), WAV 3개 0.35초 |
| 6 성능 | 전체 모델(744M)도 동작, 실행 중 최대 RAM 6.4GB / 7.3GB |
| 7 미세 조정 | accuracy@1 0.833 → 0.85, accuracy@3 0.917 → 0.967. 일반 질의 1개는 나빠짐(과적합 신호) |
| 8 실시간 검색 | yolo는 세탁기를 clock·cell phone으로 오인식. relabel이 드럼·통돌이 세탁기로 교정(3/4). yoloe만 기본 기준에서 세탁기·상자 검출 → yoloe 박스 + relabel 라벨 조합 권장 |
| 9 동작 인식 | **합성 관절**: 시간 안정화로 깜빡임 제거(만세 13→3회), 관심도 11/12, GRU는 3→6명으로 늘리자 낯선 사람 박수 0% → 95~100%. **실제 영상 8개(9-7)**: 어깨가 안 잡히는 영상(가까이 잘림·옆모습)은 판정 0%, 올림 기준 0.3→0.0으로 바꾸자 만세 63.7% → 93.4%, 교실 영상 깜빡임 21→8회. 합성으로 학습한 GRU는 실제 영상에서 확률이 0.7에 못 미쳐 쓰이지 않음 |
| 10 접근 감지 | **카메라 2대(VIGI + C920) 실제 보정**: 재투영 0.97/0.50px, 삼각측량한 칸 24.1mm(실제 24), C920 렌즈 고정값 문제를 저장된 코너로 다시 계산(RMS 0.154). **실측(대략)**: stereo 약 2.6m·높이 1.83m로 실제와 일치(계산상 오차 5.5cm), ankle1 약 20cm. **배치 비교**: 지금 배치(간격 0.38m, 높이 0.8~1.0m)는 3m에서 stereo 28cm, 1.5m만 벌리면 7cm, 안내문 권장 배치는 1cm. 낮은 카메라는 1대 추정(10-5)을 무너뜨림. 체커보드는 YOLOE 점수 0.03이라 타겟 자동 찾기 실패. 10-8은 물체가 없어 생략 |

## 실행 환경

| 항목 | 값 |
|---|---|
| 보드 | Jetson Orin Nano Super, JetPack 7.2.1, CUDA 13.2, Python 3.12, 전력 모드 MAXN_SUPER, 스왑 8GB |
| 패키지 | torch 2.14.1+cu132, sentence-transformers 6.2.0, transformers 5.20.1, ultralytics 8.4 |
| 카메라 | TP-Link VIGI C540-W (보드 유선랜 직결, RTSP, 서브 스트림 848×480), Logitech C920 USB 웹캠 (640×480 MJPG, 자동 초점 끔) — 10단계 2대 |

## 코드 (`code/`)

| 파일 | 단계 | 내용 |
|---|---|---|
| `check.py` | 2 | 설치 확인 (torch GPU, 버전) |
| `eg2.py` | 2~ | 공통 로더 + 보드용 동영상 함수(`video_frames`, `extract_frames`) |
| `test_text.py` | 3 | 텍스트 검색, 차원별 지연 |
| `capture.py`, `snap.py` | 4 | 카메라 녹화(10초 등), 사진 한 장 찍기 |
| `test_vision.py` | 4 | 이미지·동영상 검색 |
| `make_audio.py` | 5 | 경보음·잡음 생성, 카메라 마이크 녹음, 16kHz 모노 변환 |
| `test_audio.py` | 5 | 음성 검색 |
| `bench.py`, `bench_results.txt` | 6 | 5가지 구성 성능 측정 스크립트와 결과 |
| `make_data.py`, `data/` | 7 | 강의실 도메인 학습 282쌍·평가 60쌍 생성 |
| `eval_ir.py`, `train_ft.py` | 7 | 평가(Recall@k)·미세 조정 |
| `baseline_eval.txt`, `finetuned_eval.txt` | 7 | 학습 전후 평가 결과 |
| `app.py`, `run_app.sh` | 8 | 실시간 물체 검출·검색 웹앱(yolo / relabel / yoloe), 실행 스크립트 |
| `compare_photos.py`, `compare_results/` | 8 | 저장한 같은 사진으로 세 모드 비교, 모드별 결과(JSON) |
| `guard_run.sh`, `mem_guard.sh` | 8 | 보드 메모리가 400MB 아래로 떨어지면 실습 프로그램만 끄는 보호 스크립트 |
| `pose/pose_app.py`, `pose/pose_model.py`, `pose/train_pose.py` | 9 | 동작 인식 웹앱(VIGI 카메라 대응), 관절 특징·GRU, 학습 스크립트 |
| `pose/pose_sim.py`, `pose/results/` | 9 | 합성 관절 데이터 생성과 9-1~9-6 실험, 결과(JSON) |
| `pose/pose_real.py`, `pose/diag_real.py`, `pose/fix_real.py` | 9 | 9-7 실제 영상 검증: 관절 좌표만 추출(영상 저장 없음) → 규칙·GRU 판정, 오판 원인 진단, 기준값 바꾸기 실험. 결과 `pose/results/real_*.json` |
| `prox/stereo.py`, `prox/proximity_app.py` | 10 | 보정·삼각측량 공통 모듈, 접근 감지 웹앱(`--mode mono`, VIGI 입력 `--cams 0 vigi:stream2`, 카메라별 지연 `--lag`, 기본 접속 127.0.0.1) |
| `prox/camio.py` | 10 | USB 웹캠과 VIGI(RTSP)를 같은 방식으로 여는 공통 입력 (자동 초점 끔, 최근 2초 프레임 보관) |
| `prox/make_checkerboard.py` | 10-1 | 렌즈 보정용 A4 체커보드 PDF(칸 25mm, 9×6) 만들기 |
| `prox/calib_intrinsics.py` | 10-1 | 렌즈 보정 웹 도구 (코너 좌표만 저장, 덮은 범위 표시, 흔들린 장 자동 제외) |
| `prox/calib_extrinsics.py` | 10-2 | 기준점 클릭 위치 보정 웹 도구 (카메라별, 두 카메라 삼각측량 검증) |
| `prox/calib_board.py` | 10-2 | 바닥에 놓은 체커보드로 두 카메라 위치 보정 (큰 보드용, A4 보드는 2m 밖에서 코너가 안 잡힘) |
| `prox/chk10_2.py` | 10-2 | 보정 사진으로 위치 보정 검증, C920 렌즈 값별 거리 차이·거리별 정밀도 계산 |
| `prox/sim_layout.py`, `prox/target_board_test.py` | 10-3~10-7 | 실제 보정 파일로 실제 배치 거리 계산·카메라 배치 비교, 체커보드 사진으로 타겟 자동 찾기 실측. 결과 `prox/results/sim_layout.json`, `target_board.json`, `target_board_diag.txt` |
| `prox/state_logger.py` | 10-7 | 실행 중인 접근 감지 앱의 상태를 읽기만 해서 기록 (영상 없음) |
| `prox/results/compare_2cam.json`, `ankle1_live_20261102.jsonl`, `calib_README_20261102.txt` | 10-7 | 가상 2대 비교 결과, ankle1 실측 기록, 실제 보정 파일 출처 메모 |
| `prox/prox_sim.py`, `prox/results/` | 10 | 가상 강의실·보정·사람 생성과 10-6~10-8 실험(카메라 1대), 결과(JSON) |

### 실행 예

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
python test_text.py                     # 3단계
python capture.py 10                    # 카메라 10초 녹화 + 프레임 추출
python test_vision.py                   # 4단계
python bench.py full                    # 6단계 (text | vision_img | vision_vid | audio | full)
python eval_ir.py && python train_ft.py && python eval_ir.py models/eg2-classroom   # 7단계
EXTRA_LABELS=1 ./run_app.sh relabel --flip   # 8단계, 브라우저에서 :8000
cd pose && python pose_real.py extract <영상 폴더> && python pose_real.py analyze   # 9-7
cd ../prox && python proximity_app.py --cams vigi:stream2 0 --mode stereo --target-xy 1.8,0   # 10단계 (cam0=VIGI, cam1=C920)
```

## 안내문과 다르게 한 것

이 보드에서는 안내문 그대로 하면 막히는 곳이 있어서 다르게 처리했습니다. 자세한 내용은 [결과 보고서](docs/결과보고서.md)의 마지막 표에 있습니다.
- **torchaudio·torchcodec 제거**: 보드의 CUDA·ffmpeg와 맞지 않아 import가 실패합니다. 동영상은 PyAV, 음성은 librosa로 읽습니다.
- **보드 ffmpeg 기능 부족**: 녹화·RTSP·소리 생성이 안 돼서, 해당 기능을 파이썬 스크립트로 대신했습니다.
- **USB 웹캠·마이크 대신 VIGI 카메라 사용**: 앱에 `--source`, `--flip` 옵션을 추가했고, 최신 프레임만 쓰도록 했습니다.

## 올리지 않은 것

- **카메라 계정 정보**: `.camera_url`
- **카메라 사진·영상**: 사람과 이름표가 찍혀 있습니다. 9-7에 쓴 동작 영상과 10단계 보정 사진도 올리지 않았고, 관절·위치 숫자 결과만 올렸습니다.
- **모델 파일**: 미세 조정 모델, YOLO 가중치
- **로그**
