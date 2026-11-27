#!/usr/bin/env bash
# Post-deploy smoke test: does the deployed service do its main jobs, and do its safeguards hold?
#
# Usage:
#   scripts/smoke.sh https://to-ai-or-not-to-ai.onrender.com
#   AON_API_KEY=... scripts/smoke.sh https://to-ai-or-not-to-ai.onrender.com   # also checks the real key
#   scripts/smoke.sh http://localhost:8000 --rate-limit                # also proves the demo limit
#
# Environment: DEMO_KEY (default demo-public), AON_API_KEY (real key, optional),
#              SAMPLE_IMAGE (a .jpg or .png to upload; a small built-in JPEG otherwise),
#              TIMEOUT (seconds per request, default 60).
# Exit code: 0 if every check passed, 1 otherwise, so it can gate a deploy job in CI.

set -uo pipefail

BASE=""
RATE_LIMIT=0
for arg in "$@"; do
  case "$arg" in
    --rate-limit) RATE_LIMIT=1 ;;
    -h | --help) sed -n '2,13p' "$0"; exit 0 ;;
    *) BASE="${arg%/}" ;;
  esac
done
if [[ -z "$BASE" ]]; then
  echo "Usage: $0 <base-url> [--rate-limit]" >&2
  exit 2
fi

DEMO_KEY="${DEMO_KEY:-demo-public}"
REAL_KEY="${AON_API_KEY:-}"
TIMEOUT="${TIMEOUT:-60}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BODY="$TMP/body"
CODE=""
PASSED=0 FAILED=0 SKIPPED=0

if [[ -t 1 ]]; then G=$'\033[32m' R=$'\033[31m' Y=$'\033[33m' D=$'\033[2m' N=$'\033[0m'; else G="" R="" Y="" D="" N=""; fi

# request METHOD PATH [curl args...]: sets CODE and writes the response body to $BODY
request() {
  local method=$1 path=$2
  shift 2
  CODE=$(curl -sS -o "$BODY" -w '%{http_code}' --max-time "$TIMEOUT" -X "$method" "$BASE$path" "$@" 2>/dev/null) || CODE="000"
}

# expect NAME "ACCEPTED CODES" [TEXT THAT MUST APPEAR IN THE BODY]
expect() {
  local name=$1 codes=$2 needle=${3:-}
  if [[ " $codes " != *" $CODE "* ]]; then
    FAILED=$((FAILED + 1))
    printf '  %sFAIL%s  %s %s(HTTP %s, expected %s)%s\n' "$R" "$N" "$name" "$D" "$CODE" "$codes" "$N"
  elif [[ -n "$needle" ]] && ! grep -qF -- "$needle" "$BODY" 2>/dev/null; then
    FAILED=$((FAILED + 1))
    printf '  %sFAIL%s  %s %s(response has no "%s")%s\n' "$R" "$N" "$name" "$D" "$needle" "$N"
  else
    PASSED=$((PASSED + 1))
    printf '  %sPASS%s  %s\n' "$G" "$N" "$name"
  fi
}

skip() {
  SKIPPED=$((SKIPPED + 1))
  printf '  %sSKIP%s  %s %s(%s)%s\n' "$Y" "$N" "$1" "$D" "$2" "$N"
}

summary() {
  printf '\n%s passed, %s failed, %s skipped\n' "$PASSED" "$FAILED" "$SKIPPED"
}

# The image to upload: yours, or a small built-in JPEG
if [[ -n "${SAMPLE_IMAGE:-}" ]]; then
  IMG="$SAMPLE_IMAGE"
  case "$IMG" in
    *.png | *.PNG) MIME="image/png" ;;
    *) MIME="image/jpeg" ;;
  esac
else
  IMG="$TMP/sample.jpg"
  MIME="image/jpeg"
  base64 --decode > "$IMG" <<< '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDy5IfarCQ+1TpD7VYSH2r6eVQxoViukPtVhIfarCQ+1TpD7VzyqHs0KxAkPtU6Q+1WEh9qnSH2rmlUPaoViBIfarCQ+1TpD7VYSH2rmlUPZoVjmEh9qsJD7VOkPtVhIfaumVQ/HaFYgSH2qdIfarCQ+1TpD7VzyqHtUKxAkPtVhIfap0h9qsJD7VzSqHs0KxXSH2qwkPtU6Q+1WEh9q5pVD2qFY5hIfarCQ+1TpD7VYSH2rplUPx2hWIEh9qnSH2qwkPtU6Q+1c8qh7NCsQJD7VYSH2qdIfarCQ+1c0qh7VCsV0h9qsJD7VOkPtVhIfauaVQ9qhWOYSH2qwkPtU6Q+1WEh9q6ZVD8coViBIfap0h9qsJD7VOkPtXPKoe1QrECQ+1WEh9qnSH2qwkPtXNKoezQrFdIfarCQ+1TpD7VYSH2rmlUPaoVj/9k='
fi

echo "Smoke test: $BASE"

echo; echo "Server"
printf '  %s... waking the server if it sleeps (free hosting can take a minute)%s\n' "$D" "$N"
TIMEOUT_NORMAL=$TIMEOUT
TIMEOUT=120
# A sleeping host holds the connection open; a container that is still starting
# refuses it at once (HTTP 000). Retry the second case for up to a minute.
for _ in {1..30}; do
  request GET /health
  [[ "$CODE" != "000" ]] && break
  sleep 2
done
TIMEOUT=$TIMEOUT_NORMAL
expect "API is up (public health check)" 200 '"healthy"'
if [[ "$CODE" == "000" ]]; then
  printf '  %sServer unreachable, stopping here.%s\n' "$R" "$N"
  summary
  exit 1
fi
request GET /
expect "Playground page is served" 200 "To AI or Not to AI"
request GET /static/privacy.html
expect "Privacy policy is served" 200 "Privacy"
request GET /static/og.png
expect "Social preview image is served" 200

echo; echo "Inference (demo key)"
request POST /v1/inference/predict -H "X-API-Key: $DEMO_KEY" -F "file=@$IMG;type=$MIME"
expect "Fast prediction works" 200 '"prediction"'
request POST /v1/inference/explain -H "X-API-Key: $DEMO_KEY" -F "file=@$IMG;type=$MIME"
expect "GradCAM explanation works" 200 '"heatmap_base64"'

echo; echo "Feedback (demo key)"
# "UNKNOWN" is an abstention: it is stored, but the drift check ignores it,
# so smoke runs never count as votes about the model.
request POST /v1/feedback/ -H "X-API-Key: $DEMO_KEY" -H "Content-Type: application/json" \
  -d '{"filename":"smoke-test.jpg","model_prediction":"AI_GENERATED","confidence":0.5,"user_correction":"UNKNOWN","client_source":"API_v1"}'
expect "Feedback is saved" "200 201" "smoke-test.jpg"

echo; echo "Access control"
request POST /v1/inference/predict -H "X-API-Key: wrong-key" -F "file=@$IMG;type=$MIME"
expect "A wrong key is rejected" 401
request POST /v1/inference/predict -F "file=@$IMG;type=$MIME"
expect "A missing key is rejected" "401 403"
request POST /v1/admin/check-drift -H "X-API-Key: $DEMO_KEY"
expect "The demo key cannot reach admin routes" 401
if [[ -n "$REAL_KEY" ]]; then
  request GET /v1/admin/uncertain -H "X-API-Key: $REAL_KEY"
  expect "The real key reaches admin routes" 200
else
  skip "The real key reaches admin routes" "set AON_API_KEY to check it"
fi

echo; echo "Observability"
request GET /metrics
expect "Metrics endpoint responds" 200
expect "Model metrics are exported" 200 "ai_detector_predictions_total"
expect "Request metrics are exported" 200 "http_requests_total"

echo; echo "Rate limit"
if ((RATE_LIMIT)); then
  for _ in {1..40}; do
    request POST /v1/inference/predict -H "X-API-Key: $DEMO_KEY" -F "file=@$IMG;type=$MIME"
    [[ "$CODE" == "429" ]] && break
  done
  expect "The demo key is rate limited (429 within 40 calls)" 429
else
  skip "The demo key is rate limited" "add --rate-limit; it uses up your IP's demo allowance for a minute"
fi

summary
exit $((FAILED > 0))