# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for M(H)ero

block_cipher = None

a = Analysis(
    ['M(H)ero.py'],
    pathex=[],
    binaries=[],
    datas=[('Logo', 'Logo')],  # Include Logo directory
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        'matplotlib.backends.backend_qt5agg',
        'numpy',
        'pandas',
    ],
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='M(H)ero',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# For macOS, create an app bundle
app = BUNDLE(
    exe,
    name='M(H)ero.app',
    icon=None,  # Add 'icon.icns' if you create one
    bundle_identifier='com.magneticity.mhero',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'LSBackgroundOnly': 'False',
    },
)
