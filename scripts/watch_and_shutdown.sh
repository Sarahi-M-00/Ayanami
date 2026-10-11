#!/usr/bin/env bash
# Watcher: espera a que terminen las 4 descargas y apaga el equipo.
# Uso: setsid bash scripts/watch_and_shutdown.sh < /dev/null &
LOG=/tmp/watcher.log
echo "[$(date)] watcher iniciado" >> "$LOG"
while pgrep -f "fetch_weights.py|TurboFCFusion.*gguf" >/dev/null; do
  sleep 300
done
echo "[$(date)] descargas terminadas" >> "$LOG"
du -sh /home/ling/models/*/ >> "$LOG" 2>&1
# Verifica integridad mínima: cada dir con safetensors o gguf > 1GB.
ok=1
for d in qwen3-8b qwen25-coder-7b r1-distill-qwen-7b; do
  n=$(find "/home/ling/models/$d" -name '*.safetensors' 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "FALTA: $d sin safetensors" >> "$LOG"; ok=0; }
done
s=$(find /home/ling/models/qwen38-27b-turbo -name '*.gguf' -size +1G 2>/dev/null | wc -l)
[ "$s" -eq 0 ] && { echo "FALTA: turbo gguf incompleto" >> "$LOG"; ok=0; }
if [ "$ok" -eq 1 ]; then
  echo "[$(date)] todo completo, apagando" >> "$LOG"
  sudo -n poweroff >>"$LOG" 2>&1 || echo "SIN PERMISO para apagar; equipo queda encendido" >> "$LOG"
else
  echo "[$(date)] incompleto, NO apago" >> "$LOG"
fi
