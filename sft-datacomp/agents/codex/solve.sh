#!/bin/bash
set -euo pipefail

# Disable network search - work offline only
codex exec --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"