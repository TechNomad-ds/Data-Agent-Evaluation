#!/bin/bash
set -euo pipefail

container="${1:-minimal}"

export POST_TRAIN_BENCH_CONTAINERS_DIR=${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}
export APPTAINER_BIND=""

def_path="containers/${container}.def"
sif_path="${POST_TRAIN_BENCH_CONTAINERS_DIR}/${container}.sif"

if [[ ! -f "$def_path" ]]; then
  echo "ERROR: def file not found: $def_path" >&2
  exit 1
fi

apptainer build "$sif_path" "$def_path"
echo "Built: $sif_path"
