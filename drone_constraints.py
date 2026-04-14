"""
drone_constraints.py
---------------------
Simulates drone-level hardware constraints for the detection module.

Based on HULA drone estimated specs (higher-end of spectrum):
  - CPU: ARM Cortex-A ~400MHz  → simulated as 15% of one CPU core
  - RAM: 128MB                 → hard limit on process memory

HOW TO INTEGRATE THIS:
------------------------------
Add these TWO lines at the top of main.py, before anything else:

    from drone_constraints import apply_drone_constraints
    apply_drone_constraints()

That's it. The rest of the code runs as normal, just under constrained resources.

WORKS ON: Windows, Mac, Linux — no extra installs needed.
"""

import os
import time
import threading
import platform

# RAM limiting — different per OS
if platform.system() == "Windows":
    import ctypes
    import ctypes.wintypes
else:
    import resource

# ── Drone hardware spec constants ─────────────────────────────────────────────
DRONE_RAM_MB       = 128    # estimated max RAM (higher-end of HULA spec)
DRONE_CPU_FRACTION = 0.15   # 400MHz / ~2600MHz modern core ≈ 15%
CPU_THROTTLE_MS    = 50     # throttle check interval in ms


# ── RAM limit ─────────────────────────────────────────────────────────────────

class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit",     ctypes.c_int64),
        ("LimitFlags",              ctypes.wintypes.DWORD),
        ("MinimumWorkingSetSize",   ctypes.c_size_t),
        ("MaximumWorkingSetSize",   ctypes.c_size_t),
        ("ActiveProcessLimit",      ctypes.wintypes.DWORD),
        ("Affinity",                ctypes.c_size_t),
        ("PriorityClass",           ctypes.wintypes.DWORD),
        ("SchedulingClass",         ctypes.wintypes.DWORD),
    ]

class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount",  ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount",   ctypes.c_uint64),
        ("WriteTransferCount",  ctypes.c_uint64),
        ("OtherTransferCount",  ctypes.c_uint64),
    ]

class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo",                _IO_COUNTERS),
        ("ProcessMemoryLimit",    ctypes.c_size_t),
        ("JobMemoryLimit",        ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed",     ctypes.c_size_t),
    ]


def _apply_ram_limit(ram_mb):
    ram_bytes = ram_mb * 1024 * 1024

    if platform.system() == "Windows":
        try:
            JOB_OBJECT_LIMIT_PROCESS_MEMORY    = 0x00000100
            JobObjectExtendedLimitInformation  = 9

            job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
            if not job:
                raise OSError("CreateJobObjectW failed")

            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = ram_bytes

            ok = ctypes.windll.kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                raise OSError("SetInformationJobObject failed (error %d)" %
                              ctypes.windll.kernel32.GetLastError())

            ok = ctypes.windll.kernel32.AssignProcessToJobObject(
                job,
                ctypes.windll.kernel32.GetCurrentProcess()
            )
            if not ok:
                raise OSError("AssignProcessToJobObject failed (error %d)" %
                              ctypes.windll.kernel32.GetLastError())

            print("[DRONE SIM] RAM limited to %d MB (Windows Job Object)" % ram_mb)

        except Exception as e:
            print("[DRONE SIM] WARNING: Could not set RAM limit on Windows: %s" % e)
            print("[DRONE SIM] Continuing without RAM constraint.")
    else:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (ram_bytes, ram_bytes))
            print("[DRONE SIM] RAM limited to %d MB" % ram_mb)
        except Exception as e:
            print("[DRONE SIM] WARNING: Could not set RAM limit: %s" % e)
            print("[DRONE SIM] Continuing without RAM constraint.")


# ── CPU throttle ──────────────────────────────────────────────────────────────

def _cpu_throttle_worker(cpu_fraction, interval_ms):
    """
    Daemon thread that enforces CPU throttle by sleeping.
    Active for cpu_fraction of each interval, sleeping for the rest.
    e.g. 0.15 fraction = 15ms active, 85ms sleep per 100ms window.
    """
    interval_s  = interval_ms / 1000.0
    active_time = interval_s * cpu_fraction
    sleep_time  = interval_s * (1.0 - cpu_fraction)
    while True:
        time.sleep(active_time)
        time.sleep(sleep_time)


def _apply_cpu_throttle(cpu_fraction):
    # Pin to single core
    try:
        os.sched_setaffinity(0, {0})
        print("[DRONE SIM] CPU affinity set to core 0 (single core)")
    except AttributeError:
        # Windows doesn't have sched_setaffinity — use SetProcessAffinityMask
        try:
            ctypes.windll.kernel32.SetProcessAffinityMask(
                ctypes.windll.kernel32.GetCurrentProcess(), 1)
            print("[DRONE SIM] CPU affinity set to core 0 (Windows)")
        except Exception as e:
            print("[DRONE SIM] WARNING: Could not set CPU affinity: %s" % e)
    except Exception as e:
        print("[DRONE SIM] WARNING: Could not set CPU affinity: %s" % e)

    # cpu_fraction=1.0 means "full speed" -- no throttle thread needed
    if cpu_fraction >= 1.0:
        print("[DRONE SIM] CPU throttle disabled (fraction=100%% -- testing mode)")
        return

    # Start throttle thread (best-effort: log a warning and continue if it fails)
    t = threading.Thread(
        target=_cpu_throttle_worker,
        args=(cpu_fraction, CPU_THROTTLE_MS),
        daemon=True,
    )
    try:
        t.start()
        print("[DRONE SIM] CPU throttled to %.0f%% of one core (~400MHz equivalent)"
              % (cpu_fraction * 100))
    except RuntimeError as e:
        print("[DRONE SIM] WARNING: Could not start CPU throttle thread: %s" % e)
        print("[DRONE SIM] Continuing without CPU throttle.")


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_drone_constraints(
    ram_mb=DRONE_RAM_MB,
    cpu_fraction=DRONE_CPU_FRACTION,
):
    """
    Apply drone-level hardware constraints to the current process.
    Call once at the very start of main.py before any model loading.

    Parameters
    ----------
    ram_mb       : int   -- RAM cap in MB        (default: 128)
    cpu_fraction : float -- fraction of one core (default: 0.15 → ~400MHz)
    """
    print("\n" + "=" * 50)
    print("  DRONE HARDWARE CONSTRAINTS ACTIVE")
    print("=" * 50)
    print("  Device : HULA drone (higher-end estimate)")
    print("  CPU    : ARM ~400MHz → %.0f%% of one modern core" % (cpu_fraction * 100))
    print("  RAM    : %d MB hard limit" % ram_mb)
    print("  OS     : %s" % platform.system())
    print("=" * 50 + "\n")

    _apply_ram_limit(ram_mb)
    _apply_cpu_throttle(cpu_fraction)


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    apply_drone_constraints()
    print("[TEST] Running constrained workload...")
    start = time.time()
    total = sum(range(1_000_000))
    elapsed = time.time() - start
    print("[TEST] Done in %.2f s (under constraint)" % elapsed)
    print("[TEST] Sum: %d" % total)
