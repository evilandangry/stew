"""Timestamps in the training log, in seconds since this process started, to see where the time
outside the training steps goes. It only prints; nothing is timed or changed."""
import atexit
import os

_DONE = set()


def _since_start():
    try:
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return up - int(fields[19]) / os.sysconf("SC_CLK_TCK")  # field 22: start time
    except Exception:  # noqa: BLE001
        return float("nan")


def mark(label):
    if label in _DONE:
        return
    _DONE.add(label)
    print(f"[solution] timeline: {_since_start():7.2f} s  {label}", flush=True)


def install():
    mark("solution.apply() called")
    atexit.register(mark, "process exiting (training and saving done)")
