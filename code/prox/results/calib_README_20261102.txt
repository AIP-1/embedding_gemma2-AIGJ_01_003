2026-11-02 10단계 보정 결과 정리
cam0 = VIGI (vigi:stream2, 848x480)  cam0_intr.npz = vigi_intr.npz (RMS 0.727)
cam1 = C920 (/dev/video0, 640x480, MJPG, 초점 0)  cam1_intr.npz = --fix-center 값 (fx 680.6, 위치 보정에 쓴 값)
cam1_intr_free.npz = 같은 코너(0_pts.npz)로 고정 없이 다시 계산 (RMS 0.154, fx 669.9 fy 672.1, cx 328.7 cy 232.7) — 다음 위치 보정 때 사용
cam0.npz, cam1.npz, world_points.json = 체커보드 위치 보정 (16:54). 원점 = 체커보드 중심 아래 바닥, x = 카메라가 보는 방향, y = 왼쪽, z = 위
before_team_20261102/ = 그 전에 쓰던 렌즈 파일 (cam0이 C920이던 순서)
