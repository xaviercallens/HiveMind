"""Stream key events of the T4 queue: queue start/end lines, per-job progress and failures."""
import glob, os, re, time

Q = "/tmp/claude-1501372770/-home-callensxavier-gmail-com-HiveMind/22ff63cf-3b37-455e-984e-5fca864875cb/tasks/bhyjz3s3k.output"
LOGS = "/home/callensxavier_gmail_com/.claude/jobs/22ff63cf/tmp/queue_logs/"
PAT = re.compile(r"\[queue\]|wrote |Traceback|Error|error:|OutOfMemory|Killed|controls_all_passed|\"P1[12]|E13 DONE|aborted|FailFast|fail-fast|status.*failed|identity|STALL|TimeoutError|\[cell\]|cell .*(done|ok|failed)")
# start at the current end of every existing file so old events are not replayed
pos = {f: os.path.getsize(f) for f in [Q] + glob.glob(LOGS + "*.log") if os.path.exists(f)}
while True:
    for f in [Q] + glob.glob(LOGS + "*.log"):
        try:
            with open(f, "r", errors="replace") as fh:
                fh.seek(pos.get(f, 0))
                data = fh.read()
                pos[f] = fh.tell()
        except FileNotFoundError:
            continue
        for line in data.replace("\r", "\n").splitlines():
            if PAT.search(line) and "pynvml" not in line and "Loading" not in line:
                # progress lines: only every 4th window to keep the stream light
                m = re.search(r"seed (\d) window (\d+)", line)
                if m and int(m.group(2)) % 4 != 0 and "seed 1 window 0" not in line:
                    pass
                print(f"{os.path.basename(f)[:28]}: {line.strip()[:220]}", flush=True)
    time.sleep(5)
