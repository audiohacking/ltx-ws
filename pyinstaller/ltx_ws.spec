# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for LTX-WS Videofentanyl macOS app.
# Build: scripts/build_mac_app.sh

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

block_cipher = None

datas = []
web_dist = ROOT / "web" / "dist"
if web_dist.is_dir():
    datas.append((str(web_dist), "web/dist"))

hiddenimports = [
    "mlx",
    "mlx.core",
    "ltx_pipelines_mlx",
    "ltx_core_mlx",
    "huggingface_hub",
    "huggingface_hub.utils",
    "tqdm",
    "av",
    "PIL",
    "multipart",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "starlette",
    "fastapi",
    "websockets",
    "system_status",
    "ltx_paths",
    "server",
    "web_ui",
    "ltx_mlx_backend",
    "videofentanyl",
]

binaries = []

for package in ("mlx", "ltx_core_mlx", "ltx_pipelines_mlx"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:
        print(f"Warning: collect_all({package}) failed: {exc}")

a = Analysis(
    [str(ROOT / "app_main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LTX-WS-Videofentanyl",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LTX-WS-Videofentanyl",
)

app = BUNDLE(
    coll,
    name="LTX-WS Videofentanyl.app",
    icon=None,
    bundle_identifier="com.ltx-ws.videofentanyl",
    info_plist={
        "CFBundleName": "LTX-WS Videofentanyl",
        "CFBundleDisplayName": "LTX-WS Videofentanyl",
        "NSHighResolutionCapable": True,
    },
)
