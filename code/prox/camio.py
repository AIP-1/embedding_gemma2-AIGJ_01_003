"""10단계 공통 카메라 입력: USB 웹캠과 VIGI(RTSP)를 같은 방식으로 연다.

입력 이름
  0, 1 ...       USB 웹캠 /dev/videoN (MJPG, --res 크기, 자동 초점 끔)
  vigi:stream2   VIGI 서브 스트림 (848x480 고정). 과제 1의 LatestReader로 읽어 끊기면 스스로 다시 연다

주의
  - 보정(렌즈·위치)과 앱은 같은 입력 이름·같은 해상도·같은 초점으로 열어야 한다. 하나라도 바뀌면 보정을 다시 한다.
  - C920은 자동 초점이 켜져 있으면 초점이 움직일 때마다 렌즈 값이 바뀐다 → 끄고 고정 값(--focus)으로 둔다.
  - VIGI C540-W는 팬·틸트 카메라다. 위치 보정 뒤에 앱이나 버튼으로 방향을 돌리면 위치 보정을 다시 해야 한다.
  - 프레임은 디스크에 저장하지 않는다 (사람이 찍힘).
"""
import collections
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.expanduser("~/eg2-lab/cam1"))      # 과제 1의 camsrc (VIGI 계정 파일 처리 포함)


class Source:
    """최신 프레임과 최근 history_s초 프레임을 보관한다.
    lag: 이 카메라 영상이 실제보다 늦게 도착하는 시간(초). 두 카메라 프레임을 같은 순간으로 맞출 때 쓴다."""

    def __init__(self, spec, res=(640, 480), focus=0, lag=0.0, history_s=2.0):
        self.spec, self.lag = str(spec), float(lag)
        self.lock = threading.Lock()
        self.buf = collections.deque()
        self.history_s = history_s
        self.size = None
        self.n = 0
        if self.spec.isdigit():
            self.kind = "usb"
            cap = cv2.VideoCapture(int(self.spec), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
            if not cap.isOpened():
                raise RuntimeError(f"USB 카메라 {self.spec}을 열 수 없음")
            self.cap = cap
            self.focus = cap.get(cv2.CAP_PROP_FOCUS)
            threading.Thread(target=self._usb_loop, daemon=True).start()
        elif self.spec.startswith("vigi:"):
            from camsrc import LatestReader
            self.kind = "vigi"
            self.rd = LatestReader(self.spec, stale_s=3.0)
            self.focus = None
            threading.Thread(target=self._vigi_loop, daemon=True).start()
        else:
            raise ValueError(f"모르는 입력: {self.spec} (0, 1 ... 또는 vigi:stream2)")

    def _put(self, f, t):
        with self.lock:
            self.size = (f.shape[1], f.shape[0])
            self.buf.append((t, f))
            self.n += 1
            while self.buf and t - self.buf[0][0] > self.history_s:
                self.buf.popleft()

    def _usb_loop(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                self._put(f, time.time())
            else:
                time.sleep(0.05)

    def _vigi_loop(self):
        while True:
            ok, f, _, t = self.rd.read(timeout=1.0)
            if ok:
                self._put(f, t)

    @property
    def connected(self):
        return self.rd.connected if self.kind == "vigi" else True

    def latest(self):
        """(찍힌 시각 추정, 프레임) — 찍힌 시각 = 받은 시각 - lag"""
        with self.lock:
            if not self.buf:
                return None, None
            t, f = self.buf[-1]
            return t - self.lag, f

    def at(self, t_shot):
        """찍힌 시각이 t_shot에 가장 가까운 프레임: (찍힌 시각, 프레임)"""
        with self.lock:
            if not self.buf:
                return None, None
            t, f = min(self.buf, key=lambda b: abs(b[0] - self.lag - t_shot))
            return t - self.lag, f


def wait_frame(src, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        t, f = src.latest()
        if f is not None:
            return f
        time.sleep(0.05)
    raise RuntimeError(f"입력 {src.spec}에서 {timeout:.0f}초 동안 프레임이 오지 않음")


def mjpeg(get_jpeg, interval=0.07):
    """FastAPI StreamingResponse용 MJPEG 생성기"""
    while True:
        j = get_jpeg()
        if j:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
        time.sleep(interval)
