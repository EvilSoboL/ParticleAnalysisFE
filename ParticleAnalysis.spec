# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui\\main_window.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'execute',
        'execute.execute_filter',
        'execute.execute_filter.execute_sort_binarize',
        'execute.execute_analysis',
        'execute.execute_analysis.execute_ptv_analysis',
        'execute.execute_processing',
        'execute.execute_processing.vector_filter',
        'execute.execute_processing.vector_average',
        'execute.execute_processing.vector_plot',
        'execute.execute_processing.coordinate_transform',
        'execute.full_pipeline',
        'gui.automated_pipeline_tab',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ParticleAnalysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
