# REA toolchain pin (Ayanami's main RE tool) — 2026-10-13

Pinned working set on Ling's machine. If upstream ever restricts,
reinstall exactly these versions from local mirrors.

| Tool   | Version      | Source                              |
|--------|--------------|-------------------------------------|
| rea    | 6.4.0        | npm (`rea-agents`), /usr/lib/node_modules/rea-agents |
| ghidra | 12.1.2-2     | Arch extra (`/opt/ghidra`)          |
| jadx   | 1.5.6-1      | Arch extra                          |
| frida-tools | 14.11.0 | pipx (Frida core 17.23.3)          |
| jdk-openjdk | 27.u35-1 | Arch extra (needed by ghidra/jadx) |

## Offline mirror checklist (do once, store on external disk)

- [x] `npm pack rea-agents@6.4.0` tarball (in /home/ling/reserva/rea/)
- [x] Ghidra/JDK/jadx pacman pkgs (from /var/cache/pacman/pkg/, in /home/ling/reserva/rea/)
- [x] jadx (same pacman cache copy)
- [x] frida-tools + deps wheels (pip download, in /home/ling/reserva/rea/)
- [ ] Copy /home/ling/reserva/rea/ (876MB) to external disk / TeraBox

## Setup from scratch (Arch)

```bash
sudo pacman -S ghidra jadx jdk-openjdk python-pipx
sudo npm install --global rea-agents@6.4.0
pipx install frida-tools
```

## Ayanami integration plan

1. Tool-use training rows for REA commands (target → command → read evidence).
2. Scope rules: REA only on owned/authorized targets (scope.yaml).
3. R-03: REA-backed semantic verifiers first, data factory second.
