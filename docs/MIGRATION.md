# Migración a nueva laptop (Ayanami)

## Llevar (copia completa, ~87GB + este repo con .git)

- `~/Ayanami-AI/` entero (código + models/ + reserva-rea/ + student_base).
- `~/Downloads/HANDOFF_LUZ.md` (contexto personal, NO va al repo).
- Exportar de nuevo si cambia: `pacman -Qqe > docs/MIGRATION_packages.txt`.

## Destino: Arch fresco

1. `sudo pacman -S --needed - < docs/MIGRATION_packages.txt` (120 paquetes).
2. `sudo npm install --global rea-agents` + `pipx install frida-tools`.
3. Copiar `~/bin/llm`, `~/.config/hypr/`, `~/.config/fastfetch/`, `~/.bashrc`.
4. En el repo: `python3 -m venv .venv` + torch CPU + `requirements.lock`
   (ver `docs/RUNBOOK.md`), luego `.venv/bin/python -m pytest tests/ -q`.
5. Verificar: `llm -p hola` y un probe al Q4.
6. `gh` auth + `git remote -v` (el remoto sigue siendo el mismo).

## No llevar

- `.venv/` (se reconstruye), `llamita.cpp/build/` (se recompila),
  cachés (`~/.cache`), `node_modules` sueltos.
