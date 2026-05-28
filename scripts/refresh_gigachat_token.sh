#!/usr/bin/env bash
# Обновить JPY_API_TOKEN в tests/integration/.env.test через OAuth GigaChat.
#
# В WSL2 файрвол блокирует порт 9443, поэтому OAuth выполняется на
# Windows-хосте через powershell.exe (interop). Если запускать вне WSL2,
# обычный curl подойдёт.
#
# Использование:
#   GIGACHAT_AUTH_KEY=<base64> ./scripts/refresh_gigachat_token.sh
#
# где AUTH_KEY = base64(client_id:client_secret) с портала Sber DEV.
set -euo pipefail

ENV_FILE="$(dirname "$0")/../tests/integration/.env.test"
SCOPE="${GIGACHAT_SCOPE:-GIGACHAT_API_PERS}"

if [[ -z "${GIGACHAT_AUTH_KEY:-}" ]]; then
  # Попытка прочитать из существующего .env.test (если положили туда)
  if [[ -f "$ENV_FILE" ]] && grep -q '^GIGACHAT_AUTH_KEY=' "$ENV_FILE"; then
    GIGACHAT_AUTH_KEY=$(grep '^GIGACHAT_AUTH_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)
  else
    echo "ERROR: GIGACHAT_AUTH_KEY не задан и не найден в $ENV_FILE" >&2
    echo "Положите его как: GIGACHAT_AUTH_KEY=<base64> или в .env.test строкой" >&2
    exit 1
  fi
fi

REQ_UID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen)

# Если powershell.exe доступен — идём через Windows-хост (обход WSL-файрвола).
if command -v powershell.exe >/dev/null 2>&1; then
  echo "[refresh-token] using powershell.exe (WSL host bypass)"
  RESPONSE=$(powershell.exe -NoProfile -Command "
    [Net.ServicePointManager]::ServerCertificateValidationCallback = {\$true};
    try {
      \$r = Invoke-RestMethod -Uri 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth' \
        -Method Post \
        -Headers @{'Authorization'='Basic ${GIGACHAT_AUTH_KEY}'; 'RqUID'='${REQ_UID}'; 'Content-Type'='application/x-www-form-urlencoded'} \
        -Body 'scope=${SCOPE}' \
        -TimeoutSec 20;
      \$r | ConvertTo-Json -Compress
    } catch { Write-Error \$_.Exception.Message; exit 1 }
  " 2>&1)
else
  echo "[refresh-token] using local curl"
  RESPONSE=$(curl -k -s --max-time 20 \
    -X POST 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth' \
    -H "Authorization: Basic ${GIGACHAT_AUTH_KEY}" \
    -H "RqUID: ${REQ_UID}" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d "scope=${SCOPE}")
fi

# Очистка CRLF от powershell
RESPONSE=$(printf '%s' "$RESPONSE" | tr -d '\r')

# Грубый JSON-парс access_token
TOKEN=$(printf '%s' "$RESPONSE" | python3 -c '
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print(data["access_token"])
except Exception as e:
    print("PARSE_ERROR:", e, file=sys.stderr)
    sys.exit(1)
')

if [[ -z "$TOKEN" || "$TOKEN" == PARSE_ERROR* ]]; then
  echo "[refresh-token] не удалось распарсить ответ:" >&2
  echo "$RESPONSE" >&2
  exit 1
fi

# Обновить .env.test (создать если нет)
if [[ ! -f "$ENV_FILE" ]]; then
  mkdir -p "$(dirname "$ENV_FILE")"
  cp "$(dirname "$ENV_FILE")/.env.test.example" "$ENV_FILE" 2>/dev/null || true
fi

if grep -q '^JPY_API_TOKEN=' "$ENV_FILE" 2>/dev/null; then
  # На месте заменить (через python чтобы избежать sed-эскейпов в длинном токене)
  ENV_FILE="$ENV_FILE" TOKEN="$TOKEN" python3 -c "
import os, pathlib
p = pathlib.Path(os.environ['ENV_FILE'])
tok = os.environ['TOKEN']
lines = p.read_text().splitlines()
for i, l in enumerate(lines):
    if l.startswith('JPY_API_TOKEN='):
        lines[i] = f'JPY_API_TOKEN={tok}'
        break
p.write_text('\n'.join(lines) + '\n')
"
else
  echo "JPY_API_TOKEN=$TOKEN" >> "$ENV_FILE"
fi

echo "[refresh-token] OK — $ENV_FILE updated (token ${TOKEN:0:30}…)"
