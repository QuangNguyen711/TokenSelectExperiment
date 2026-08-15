# File: patcher/prefill_profiler.py
"""
Do breakdown thoi gian prefill cua TokenSelect theo tung giai doan.

Vi sao dung wall-clock + cuda.synchronize() thay vi CUDA event:
phan lon chi phi o day nam o CPU (vong lap Python dung chunk-plan, .item() gay
sync, dung lai flashinfer wrapper moi sub-chunk). CUDA event chi do timeline GPU
nen se bo sot toan bo phan do. Doi lai, moi lan sync ton ~10-20us; baseline goi
nhieu hon SCR ~8 lan nen chiu overhead nhieu hon -> luon chay kem mot lan
ENABLED=False de tru phan nay ra truoc khi ket luan.

Tat hoan toan khi ENABLED=False (khong ton gi).
"""
import json
import os
import time

import torch

ENABLED = False
OUTPUT_PATH = None
PROFILE_LAYER = 0   # chi do 1 layer, nhan len khi phan tich
N_LAYERS = 28

_totals = {}      # stage -> tong giay (cong don toan run)
_counts = {}      # counter -> tong
_samples = 0
_per_sample = []  # [{stage: giay, ...}] moi phan tu = 1 request -> de tinh mean+-std
_last_snap = {}
_n_timer_calls = 0   # so lan bam gio -> dung de tru overhead cua chinh bo do


def enable(flag=True, output_path=None):
    global ENABLED, OUTPUT_PATH
    ENABLED = flag
    if output_path:
        OUTPUT_PATH = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if flag:
        # dump() thuong chi chay khi co request moi -> sample cuoi cung se mat.
        import atexit
        atexit.register(dump)


class stage:
    """with prof.stage('3_retrieval', _prof): ...   (on=False -> khong lam gi)"""

    __slots__ = ("name", "on", "t0")

    def __init__(self, name, on=True):
        self.name = name
        self.on = on
        self.t0 = 0.0

    def __enter__(self):
        if self.on:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.on:
            global _n_timer_calls
            torch.cuda.synchronize()
            _totals[self.name] = _totals.get(self.name, 0.0) + (time.perf_counter() - self.t0)
            _counts["ncall_" + self.name] = _counts.get("ncall_" + self.name, 0) + 1
            _n_timer_calls += 1
        return False


def add(name, seconds):
    global _n_timer_calls
    _totals[name] = _totals.get(name, 0.0) + seconds
    _counts["ncall_" + name] = _counts.get("ncall_" + name, 0) + 1
    _n_timer_calls += 1


def count(name, n=1):
    _counts[name] = _counts.get(name, 0) + n


def next_sample():
    """Goi khi bat dau mot request moi -> chot so lieu cua request vua xong."""
    global _samples, _last_snap
    if _totals:
        delta = {k: v - _last_snap.get(k, 0.0) for k, v in _totals.items()}
        if any(v > 0 for v in delta.values()):
            _per_sample.append(delta)
        _last_snap = dict(_totals)
    _samples += 1


def dump():
    # Server chay o tien trinh con (fork) nen ca cha lan con deu co atexit.
    # Tien trinh cha khong do gi ca; neu de no ghi thi no de len ket qua that
    # cua con. Tach file theo PID + bo qua dump rong de tranh han.
    if not OUTPUT_PATH or not _totals:
        return
    base, ext = os.path.splitext(OUTPUT_PATH)
    with open(f"{base}.pid{os.getpid()}{ext}", "w", encoding="utf-8") as f:
        json.dump({"samples": _samples, "stages": _totals, "counts": _counts,
                   "profile_layer": PROFILE_LAYER, "n_layers": N_LAYERS,
                   "n_timer_calls": _n_timer_calls, "per_sample": _per_sample},
                  f, indent=2)


def reset():
    global _samples
    _totals.clear()
    _counts.clear()
    _samples = 0
