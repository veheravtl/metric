#!/usr/bin/env bash
#
# Устанавливает фиксированный portable Blender без sudo и без изменения PATH.
# Бинарные файлы остаются в .tools/, который исключён из Git.

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/.." && pwd)"
TOOLS_DIRECTORY="${PROJECT_ROOT}/.tools"
BLENDER_VERSION="4.5.14"
ARCHIVE_NAME="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
ARCHIVE_PATH="${TOOLS_DIRECTORY}/${ARCHIVE_NAME}"
INSTALL_DIRECTORY="${TOOLS_DIRECTORY}/blender-${BLENDER_VERSION}-linux-x64"
DOWNLOAD_URL="https://download.blender.org/release/Blender4.5/${ARCHIVE_NAME}"
EXPECTED_SHA256="9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"

if [[ -x "${INSTALL_DIRECTORY}/blender" ]]; then
    "${INSTALL_DIRECTORY}/blender" --version | head -n 1
    echo "Blender уже установлен: ${INSTALL_DIRECTORY}"
    exit 0
fi

mkdir -p "${TOOLS_DIRECTORY}"
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Скачивание ${DOWNLOAD_URL}"
    curl --fail --location "${DOWNLOAD_URL}" --output "${ARCHIVE_PATH}"
fi

printf '%s  %s\n' "${EXPECTED_SHA256}" "${ARCHIVE_PATH}" | sha256sum --check -
tar -xJf "${ARCHIVE_PATH}" -C "${TOOLS_DIRECTORY}"
"${INSTALL_DIRECTORY}/blender" --version | head -n 1
echo "Blender установлен: ${INSTALL_DIRECTORY}"
