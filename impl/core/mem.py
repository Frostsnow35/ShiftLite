"""进程内存采样（跨平台，零新增依赖）。

口径：进程 RSS（Resident Set Size），单位 MB。

- Windows：psapi.GetProcessMemoryInfo 的 WorkingSetSize；
- Linux / macOS：/proc/self/statm 的驻留页数 × 页大小；
- 都拿不到时返回 None，调用方如实标注"未采集"，不猜、不填 0。

RSS 是**整个进程**的量，不是单个模型的量。同一进程里顺序加载多套运行时时，
后测组别的绝对值会包含先加载模型的常驻，因此 README 里两个口径分开记：
绝对值只作本机参考，跨组横向比较看组内增量（推理期间峰值 − 推理前基线）。
"""

from __future__ import annotations

import sys
from ctypes import byref, c_ulong, c_size_t, c_void_p, sizeof, windll
from ctypes import Structure


def rss_mb() -> float | None:
    """当前进程 RSS（MB）。取不到返回 None。"""
    if sys.platform.startswith("win"):
        return _win_rss_mb()
    return _posix_rss_mb()


def _posix_rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            fields = f.read().split()
        pages = int(fields[1])
        return pages * _page_size() / (1024 * 1024)
    except Exception:
        return None


def _page_size() -> int:
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 4096


class _PROCESS_MEMORY_COUNTERS(Structure):
    # 必须用 c_size_t：Windows 是 LLP64（unsigned long 只 32 位），用 c_ulong
    # 会让 64 位上的 SIZE_T 字段集体错位，GetProcessMemoryInfo 直接返回失败。
    _SZ = c_size_t
    _fields_ = [
        ("cb", c_ulong),
        ("PageFaultCount", c_ulong),
        ("PeakWorkingSetSize", _SZ),
        ("WorkingSetSize", _SZ),
        ("QuotaPeakPagedPoolUsage", _SZ),
        ("QuotaPagedPoolUsage", _SZ),
        ("QuotaPeakNonPagedPoolUsage", _SZ),
        ("QuotaNonPagedPoolUsage", _SZ),
        ("PagefileUsage", _SZ),
        ("PeakPagefileUsage", _SZ),
    ]


def _win_rss_mb() -> float | None:
    try:
        kernel32 = windll.kernel32
        # 不声明 restype 时 ctypes 按 32 位 int 返回，64 位进程句柄会被截断，
        # GetProcessMemoryInfo 拿到无效句柄静默失败（返回 0 且 GetLastError=0）。
        kernel32.GetCurrentProcess.restype = c_void_p
        kernel32.GetCurrentProcess.argtypes = []

        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = sizeof(pmc)
        for dll_name in ("psapi", "kernel32"):   # Win7 之后两者都导出该函数
            try:
                dll = windll.LoadLibrary(dll_name) if dll_name == "psapi" \
                    else kernel32
                dll.GetProcessMemoryInfo.argtypes = [c_void_p, c_void_p, c_ulong]
                dll.GetProcessMemoryInfo.restype = c_ulong
                ok = dll.GetProcessMemoryInfo(
                    kernel32.GetCurrentProcess(), byref(pmc), sizeof(pmc))
            except Exception:
                continue
            if ok:
                return pmc.WorkingSetSize / (1024 * 1024)
        return None
    except Exception:
        return None


def self_test() -> None:
    v = rss_mb()
    if v is None:
        print("RSS 采样不可用（本平台未支持）——调用方会标注『未采集』")
        return
    big = bytearray(64 * 1024 * 1024)   # 申请并触碰 64 MB，RSS 应上升
    for i in range(0, len(big), 4096):
        big[i] = 1
    after = rss_mb()
    print(f"RSS 采样可用：基线 {v:.1f} MB → 申请 64 MB 后 {after:.1f} MB "
          f"（Δ {after - v:+.1f} MB，应约为 +64）")
    del big


__all__ = ["rss_mb", "self_test"]

if __name__ == "__main__":
    self_test()
