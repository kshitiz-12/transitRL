#!/usr/bin/env bash

set -uo pipefail

PING_URL="${1:-}"

if [ -z "$PING_URL" ]; then
  echo "Usage: bash validate.sh <ping_url>"
  exit 1
fi

PING_URL="${PING_URL%/}"

echo ""
echo "========================================"
echo "  OpenEnv Submission Validator"
echo "========================================"
echo "Ping URL: $PING_URL"
echo ""

# ---------------- STEP 1 ----------------
echo "Step 1/3: Checking HF Space..."

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset")

if [ "$HTTP_CODE" = "200" ]; then
  echo "PASSED -- HF Space is live and /reset works"
else
  echo "FAILED -- HF Space /reset returned $HTTP_CODE"
  exit 1
fi

# ---------------- STEP 2 ----------------
echo ""
echo "Step 2/3: Checking Docker build..."

if docker build . >/dev/null 2>&1; then
  echo "PASSED -- Docker build succeeded"
else
  echo "Retrying using PowerShell..."
  powershell.exe -Command "docker build ." >/dev/null 2>&1

  if [ $? -eq 0 ]; then
    echo "PASSED -- Docker build succeeded (via PowerShell)"
  else
    echo "FAILED -- Docker build failed"
    exit 1
  fi
fi

# ---------------- STEP 3 ----------------
echo ""
echo "Step 3/3: Checking openenv validation..."

echo "Detecting openenv path..."

# Try to find openenv.exe automatically
OPENENV_PATH=$(powershell.exe -Command "(Get-Command openenv).Source" 2>/dev/null | tr -d '\r')

if [ -z "$OPENENV_PATH" ]; then
  echo "FAILED -- openenv not found in system"
  exit 1
fi

echo "Found openenv at: $OPENENV_PATH"

# Run validation via PowerShell (safe)
powershell.exe -Command "& '$OPENENV_PATH' validate"

if [ $? -eq 0 ]; then
  echo "PASSED -- openenv validate passed"
else
  echo "FAILED -- openenv validate failed"
  exit 1
fi

# ---------------- DONE ----------------
echo ""
echo "========================================"
echo "ALL CHECKS PASSED 💀"
echo "YOUR SUBMISSION IS READY 🚀"
echo "========================================"