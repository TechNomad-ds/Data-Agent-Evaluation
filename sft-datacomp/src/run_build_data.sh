#!/bin/bash
set -euo pipefail

AGENT="${1:-claude}"
AGENT_CONFIG="${2:-claude-3-5-sonnet-20241022}"
INPUT_MD="${3:-}"   # markdown文件路径

if [[ -z "${INPUT_MD}" ]]; then
  echo "ERROR: input markdown path is required."
  echo "Usage: bash src/run_build_data.sh <agent> <agent_config> <path/to/book.md>"
  exit 1
fi

if [[ ! -f "${INPUT_MD}" ]]; then
  echo "ERROR: input markdown file not found: ${INPUT_MD}"
  exit 1
fi

case "${INPUT_MD}" in
  *.md|*.markdown) ;;
  *)
    echo "ERROR: input file must be a Markdown file (.md/.markdown): ${INPUT_MD}"
    exit 1
    ;;
esac

echo "Running training data construction task (Markdown -> JSONL)"
echo "Agent: ${AGENT}"
echo "Agent Config: ${AGENT_CONFIG}"
echo "Input Markdown: ${INPUT_MD}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

SAFE_AGENT_CONFIG="$(echo "${AGENT_CONFIG}" | sed 's#[^a-zA-Z0-9._-]#_#g')"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${REPO_ROOT}/results/${AGENT}_${SAFE_AGENT_CONFIG}_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"
echo "Results will be saved to: ${RESULT_DIR}"

PROMPT="$(python3 "${SCRIPT_DIR}/get_prompt.py" --agent "${AGENT}")"
echo "${PROMPT}" > "${RESULT_DIR}/prompt.txt"

TMP_DIR="/tmp/data_build_${TIMESTAMP}"
JOB_DIR="${TMP_DIR}/job_dir"
JOB_TMP="${TMP_DIR}/tmp"

mkdir -p \
  "${JOB_DIR}/task/source_data/default" \
  "${JOB_DIR}/task/training_data" \
  "${JOB_TMP}"

cp "${INPUT_MD}" "${JOB_DIR}/task/source_data/default/book.md"

if [[ ! -f "${REPO_ROOT}/agents/${AGENT}/solve.sh" ]]; then
  echo "ERROR: Agent solve.sh not found: agents/${AGENT}/solve.sh" >&2
  exit 1
fi
cp "${REPO_ROOT}/agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"
chmod +x "${JOB_DIR}/agent_solve.sh"

# Choose container
CONTAINER_SIF="${REPO_ROOT}/containers/minimal.sif"
if [[ ! -f "${CONTAINER_SIF}" ]]; then
  echo "ERROR: Container not found: ${CONTAINER_SIF}" >&2
  echo "Hint: build it first: bash containers/build_container.sh minimal" >&2
  exit 1
fi
echo "Using container: ${CONTAINER_SIF}"

echo "Running agent in container..."

apptainer exec \
  --env PYTHONNOUSERSITE="1" \
  --env PROMPT="${PROMPT}" \
  --env AGENT_CONFIG="${AGENT_CONFIG}" \
  --bind "${JOB_TMP}:/tmp" \
  --home "${JOB_DIR}:/home/ben" \
  --pwd "/home/ben/task" \
  --writable-tmpfs \
  "${CONTAINER_SIF}" \
  bash "/home/ben/agent_solve.sh" > "${RESULT_DIR}/agent_output.txt" 2>&1

echo "Agent execution completed."

# Hard fail if missing required artifact
if [[ ! -f "${JOB_DIR}/task/training_data/train.jsonl" ]]; then
  echo "ERROR: agent did not produce task/training_data/train.jsonl" >&2
  echo "See agent log: ${RESULT_DIR}/agent_output.txt" >&2
  cp -r "${JOB_DIR}/task" "${RESULT_DIR}/task_snapshot" || true
  exit 2
fi

# Save outputs
cp -r "${JOB_DIR}/task/training_data" "${RESULT_DIR}/"
cp -r "${JOB_DIR}/task" "${RESULT_DIR}/task_snapshot"

echo "Evaluating training data..."
python3 "${SCRIPT_DIR}/evaluate_training_data.py" \
  --model-path "${RESULT_DIR}/task_snapshot" \
  --json-output-file "${RESULT_DIR}/metrics.json" > "${RESULT_DIR}/evaluation_output.txt" || true

echo "Task completed!"
echo "Results saved to: ${RESULT_DIR}"

rm -rf "${TMP_DIR}"
echo "Done."
