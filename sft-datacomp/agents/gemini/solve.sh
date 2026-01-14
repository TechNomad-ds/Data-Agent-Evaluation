#!/bin/bash
set -euo pipefail

export GEMINI_SANDBOX="false"
gemini --yolo --model "$AGENT_CONFIG" --output-format stream-json -p "$PROMPT"