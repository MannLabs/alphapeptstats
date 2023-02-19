# -*- mode: python ; coding: utf-8 -*-

import pkgutil
import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE, TOC
import PyInstaller.utils.hooks
import pkg_resources
import importlib.metadata
import alphastats


##################### User definitions
exe_name = 'alphastats_gui'
script_name = 'alphastats_pyinstaller.py'
if sys.platform[:6] == "darwin":
	icon = '../logos/alphapeptstats_logo.icns'
else:
	icon = '../logos/alphapeptstats_logo.ico'
block_cipher = None
location = os.getcwd()
project = "alphastats"
remove_tests = True
bundle_name = "alphastats"
#####################
block_cipher = None


a = Analysis(
    ['alphastats_pyinstaller.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


pyz = PYZ(
	a.pure,
	a.zipped_data,
	cipher=block_cipher
)

if sys.platform[:5] == "linux":
	exe = EXE(
		pyz,
		a.scripts,
		a.binaries,
		a.zipfiles,
		a.datas,
		name=bundle_name,
		debug=False,
		bootloader_ignore_signals=False,
		strip=False,
		upx=True,
		console=True,
		upx_exclude=[],
		icon=icon
	)
else:
	exe = EXE(
		pyz,
		a.scripts,
		# a.binaries,
		a.zipfiles,
		# a.datas,
		exclude_binaries=True,
		name=exe_name,
		debug=False,
		bootloader_ignore_signals=False,
		strip=False,
		upx=True,
		console=True,
		icon=icon
	)
	coll = COLLECT(
		exe,
		a.binaries,
		# a.zipfiles,
		a.datas,
		strip=False,
		upx=True,
		upx_exclude=[],
		name=exe_name
	)
	if sys.platform[:6] == "darwin":
		import cmath
		import shutil
		shutil.copyfile(
			cmath.__file__,
			f"dist/{exe_name}/{os.path.basename(cmath.__file__)}"
		)
