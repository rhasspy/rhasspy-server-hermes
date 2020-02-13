# -*- mode: python -*-
import os
import site
from pathlib import Path

site_dirs = site.getsitepackages()

# Add explicit site packages
rhasspy_site_packages = os.environ.get("RHASSPY_SITE_PACKAGES")
if rhasspy_site_packages:
    site_dirs = [rhasspy_site_packages] + site_dirs

# Add virtual environment site packages
venv_path = os.environ.get("VIRTUAL_ENV")
if venv_path:
    venv_lib = Path(venv_path) / "lib"
    for venv_python_dir in venv_lib.glob("python*"):
        venv_site_dir = venv_python_dir / "site-packages"
        if venv_site_dir.is_dir():
            site_dirs.append(venv_site_dir)

# Look for compiled artifacts
rhasspyprofile_path = None
for site_dir in site_dirs:
    site_dir = Path(site_dir)
    for package_dir in site_dir.glob("*"):
        if package_dir.is_dir() and package_dir.name == "rhasspyprofile":
            rhasspyprofile_path = package_dir
            break

assert rhasspyprofile_path, "Need path to rhasspyprofile module"

block_cipher = None

a = Analysis(
    [Path.cwd() / "__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[
        (f, str("rhasspyprofile" / f.relative_to(rhasspyprofile_path).parent))
        for f in rhasspyprofile_path.glob("profiles/**/*")
    ],
    hiddenimports=["pkg_resources.py2_warn"],
    hookspath=[],
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
    name="rhasspyserver_hermes",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="rhasspyserver_hermes",
)
