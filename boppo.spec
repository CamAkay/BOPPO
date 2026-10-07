from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

a = Analysis(
    ['boppo.py'],
    pathex=[],
    binaries=[],
    datas=[
        *collect_data_files('openpyxl'),
        *collect_data_files('docx'),  # python-docx default template, for the metrics .docx
        *collect_data_files('matplotlib'),
    ],
    hiddenimports=[
        'PyQt5.QtPrintSupport',
        'PyQt5.sip',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_agg',
        'numpy.lib.format',
        'docx',
    ],
    excludes=['psse3605', 'psspy', 'dyntools', 'mhi'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='BOPPO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='boppo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BOPPO',
)
