#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILLS_SRC="${REPO_ROOT}/skills"
SKILLS_DST="${HOME}/.codex/skills/pto-kernels"

mkdir -p "${SKILLS_DST}"

for skill_dir in "${SKILLS_SRC}"/*; do
  [[ -d "${skill_dir}" ]] || continue
  if [[ ! -f "${skill_dir}/SKILL.md" ]]; then
    echo "[WARN] Skipping ${skill_dir}: missing SKILL.md" >&2
    continue
  fi
  rm -rf "${SKILLS_DST}/$(basename "${skill_dir}")"
  cp -R "${skill_dir}" "${SKILLS_DST}/"
  echo "[skills] installed $(basename "${skill_dir}")"
done

echo "[skills] destination: ${SKILLS_DST}"
