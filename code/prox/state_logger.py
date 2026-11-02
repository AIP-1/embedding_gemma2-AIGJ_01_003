"""팀원 proximity_app의 /state를 읽기만 해서 기록 (영상은 받지 않음)
python state_logger.py <포트> <출력.jsonl> [최대 초]
- 0.5초마다 stats·people·target을, 새 이벤트는 한 번씩 기록
- 실행 중인 proximity_app 명령줄(모드·옵션)이 바뀌면 기록
- 앱이 30분 넘게 없으면 끝냄
"""
import json, subprocess, sys, time, urllib.request
port, out = int(sys.argv[1]), sys.argv[2]
max_s = float(sys.argv[3]) if len(sys.argv) > 3 else 3 * 3600
t0, last_ok, seen, last_cmd = time.time(), time.time(), set(), None
with open(out, "a") as f:
    while time.time() - t0 < max_s and time.time() - last_ok < 1800:
        now = time.time()
        try:
            cmd = subprocess.run(["pgrep", "-af", "[p]roximity_app.py"], capture_output=True, text=True).stdout.strip()
        except Exception:
            cmd = ""
        if cmd != last_cmd:
            f.write(json.dumps({"t": now, "kind": "cmd", "cmd": cmd}, ensure_ascii=False) + "\n"); last_cmd = cmd
        try:
            s = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2))
            last_ok = now
            f.write(json.dumps({"t": now, "kind": "state", "stats": s.get("stats"), "people": s.get("people"),
                                "target": s.get("target"), "near": s.get("near"), "far": s.get("far"),
                                "cams": s.get("cams"), "moving": s.get("moving"), "cam_pos": s.get("cam_pos")},
                               ensure_ascii=False) + "\n")
            for e in s.get("events", []):
                k = (round(e["ts"], 3), e["who"], e["label"])
                if k not in seen:
                    seen.add(k)
                    f.write(json.dumps({"t": now, "kind": "event", **e}, ensure_ascii=False) + "\n")
        except Exception as ex:
            pass
        f.flush()
        time.sleep(0.5)
