# Machine inventory — Phase 0 (measured 2026-10-03, America/Guatemala)
# Source: direct command output. No values invented.

## Machine
- Product: HP EliteBook 845 G8 Notebook PC (`/sys/class/dmi/id/product_name`)
- Vendor: HP
- Matches prompt expectation "HP 845 G8 laptop with 16 GB RAM": yes (14.9 GiB usable, see RAM)

## OS
- Arch Linux (rolling), `PRETTY_NAME="Arch Linux"`
- Kernel: `Linux Ling 7.2.8-zen1-2-zen #1 ZEN SMP PREEMPT_DYNAMIC Thu, 01 Oct 2026 18:14:15 +0000 x86_64 GNU/Linux`
- WSL2 note: not applicable (native Linux)

## CPU (`lscpu`)
- Architecture: x86_64
- Model: AMD Ryzen 5 PRO 5650U with Radeon Graphics
- Sockets: 1, Cores per socket: 6, Threads per core: 2 → 12 logical CPUs
- Frequency at measure time: 54% scaling (min 413 MHz, max 2301 MHz)
- SIMD relevant flags present: avx, avx2, fma, f16c, bmi2
- NOT present: avx512*, amx_* (no AVX-512, no AMX) → bf16/fp32 vector throughput is AVX2-class only

## RAM (`free -h`, `/proc/meminfo`)
- MemTotal: 15653768 kB (~14.9 GiB)
- At measure time: total 14Gi, used 2.3Gi, free 11Gi, available ~12Gi (varies with desktop load)
- Swap: 4.0Gi total, 0B used (zram)
- Implication: the full ~3.4 GB BF16 student fits in RAM for inference, but training (AdamW states + activations) does not fit comfortably; see compute branch decision

## Disk (`df -h`)
- /home (/dev/nvme0n1p3): 202G total, 41G used, 151G available (22%)
- /tmp (tmpfs): 7.5G total, 7.4G available
- Space is not a constraint for this project

## Python (system)
- `python3 -V`: Python 3.14.7 (`/usr/bin/python3`)
- Note: Phase 2 requires a venv with Python 3.10–3.12 for the torch/transformers stack

## GPU
- `nvidia-smi`: not installed, no NVIDIA hardware
- `rocm-smi`: not installed, no ROCm stack
- PCI VGA: `04:00.0 Advanced Micro Devices, Inc. [AMD/ATI] Cezanne [Radeon Vega Series / Radeon Vega Mobile Series] (rev d2)` — integrated GPU only
- Conclusion: **no usable GPU for PyTorch** (no CUDA, no ROCm). CPU-only torch is the only local option.

## Raw command outputs (verbatim, 2026-10-03)
```
$ uname -a
Linux Ling 7.2.8-zen1-2-zen #1 ZEN SMP PREEMPT_DYNAMIC Thu, 01 Oct 2026 18:14:15 +0000 x86_64 GNU/Linux
$ free -h
               total        used        free      shared  buff/cache   available
Mem:            14Gi       2.3Gi        11Gi       119Mi       2.0Gi        12Gi
Swap:          4.0Gi          0B       4.0Gi
$ df -h /home/ling /tmp
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p3  202G   41G  151G  22% /home
tmpfs           7.5G  114M  7.4G   2% /tmp
$ python3 -V
Python 3.14.7
```
