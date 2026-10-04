#!/usr/bin/env bash
# Единая точка входа в зафиксированную проектом версию Blender.

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
BLENDER_EXECUTABLE="${PROJECT_ROOT}/.tools/blender-4.5.14-linux-x64/blender"

if [[ ! -x "${BLENDER_EXECUTABLE}" ]]; then
    echo "Blender не найден. Сначала выполните: ./scripts/install_blender.sh" >&2
    exit 1
fi

exec "${BLENDER_EXECUTABLE}" "$@"
