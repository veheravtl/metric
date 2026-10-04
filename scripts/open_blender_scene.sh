#!/usr/bin/env bash
# Открывает готовую G0-сцену в графическом интерфейсе Blender.

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
SCENE_PATH="${1:-${PROJECT_ROOT}/outputs/synthetic_3d/g0_smoke/scene.blend}"

if [[ ! -f "${SCENE_PATH}" ]]; then
    echo "Сцена не найдена: ${SCENE_PATH}" >&2
    echo "Сначала выполните smoke-тест по инструкции docs/blender-quickstart.md" >&2
    exit 1
fi

# На проверенной Ubuntu-сессии нативный Wayland не нашёл libdecor и открыл окно
# без системной рамки. Если XWayland уже доступен через DISPLAY, выбираем его
# только для интерактивного viewer-а. Пакетный рендер это не затрагивает.
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" && -n "${DISPLAY:-}" ]]; then
    export WAYLAND_DISPLAY=""
fi

exec "${SCRIPT_DIRECTORY}/blender.sh" "${SCENE_PATH}"
