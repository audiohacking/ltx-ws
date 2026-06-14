#!/usr/bin/env bash
# Install Python deps for PyInstaller macOS app builds (local or GitHub Actions).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${LTX2_MLX_GIT_TAG:-v0.14.9}"

echo "==> Python: $(python3 --version)"
echo "==> LTX-2-MLX tag: ${TAG}"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-build.txt
python3 -m pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@${TAG}#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@${TAG}#subdirectory=packages/ltx-pipelines-mlx"

python3 -c "import mlx; import ltx_pipelines_mlx; import PyInstaller; print('build deps OK')"
