@echo off
setlocal
title Kimodo install
:: Kimodo installer (Windows, NVIDIA GPU). Safe to run again: every step skips what is
:: already in place, and nothing that works is uninstalled or overwritten.
:: Layout (all next to this file):
::   venv311\                    Python 3.11 venv, packages pinned by requirements-lock.txt
::   KIMODO-Meta3_llm2vec_NF4\   NF4 text encoder (Aero-Ex)
::   kimodo\                     alivnal/kimodo (fork of Aero-Ex/kimodo), branch "local"; kimodo\kimodo-viser\ inside
::   wheels\                     prebuilt motion_correction wheel
::   scripts\kimodo_check.py     end-to-end check, run as the last step

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "VENV=%ROOT%\venv311"
set "PY=%VENV%\Scripts\python.exe"
set "REPO=%ROOT%\kimodo"
set "NF4=%ROOT%\KIMODO-Meta3_llm2vec_NF4"
set "VISER_CLIENT=%REPO%\kimodo-viser\src\viser\client"
set "MC_WHEEL=%ROOT%\wheels\motion_correction-1.0.0-cp311-cp311-win_amd64.whl"
set "MC_URL=https://github.com/Aero-Ex/kimodo/releases/download/v1.0.0/motion_correction-1.0.0-cp311-cp311-win_amd64.whl"

echo ============================================
echo   Kimodo installer
echo   %ROOT%
echo ============================================
echo.

:: ---------------------------------------------------------------- 1. prerequisites
echo [1/8] Prerequisites
py -3.11 --version >nul 2>&1
if errorlevel 1 call :fail "Python 3.11 not found. Install Python 3.11 so that 'py -3.11' works." & exit /b 1
where git >nul 2>&1
if errorlevel 1 call :fail "Git not found on PATH." & exit /b 1
where nvidia-smi >nul 2>&1
if errorlevel 1 call :fail "nvidia-smi not found. Kimodo needs an NVIDIA GPU and driver." & exit /b 1
for /f "tokens=2" %%v in ('py -3.11 --version 2^>^&1') do echo   Python %%v, git and NVIDIA driver found.
echo.

:: ---------------------------------------------------------------- 2. venv
echo [2/8] Virtual environment (venv311)
if exist "%PY%" (
    echo   exists, kept.
) else (
    py -3.11 -m venv "%VENV%"
    if errorlevel 1 call :fail "Could not create venv311." & exit /b 1
    echo   created.
)
echo.

:: ---------------------------------------------------------------- 3. pinned packages
echo [3/8] Pinned packages (requirements-lock.txt)
:: Installs only what is missing or at another version; torch stays the cu128 build.
:: pip is skipped when the venv already matches the lock (pip would re-clone the git pins).
"%PY%" "%ROOT%\scripts\lock_satisfied.py"
if errorlevel 1 (
    "%PY%" -m pip install --disable-pip-version-check -q -r "%ROOT%\requirements-lock.txt"
    if errorlevel 1 call :fail "pip could not install requirements-lock.txt." & exit /b 1
    echo   installed.
) else (
    echo   venv matches the lock already.
)
echo.

:: ---------------------------------------------------------------- 4. NF4 text encoder
echo [4/8] NF4 text encoder
if exist "%NF4%\model.safetensors" (
    echo   present, kept.
) else (
    "%PY%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aero-Ex/KIMODO-Meta3_llm2vec_NF4', local_dir=r'%NF4%')"
    if errorlevel 1 call :fail "Could not download Aero-Ex/KIMODO-Meta3_llm2vec_NF4." & exit /b 1
    echo   downloaded.
)
echo.

:: ---------------------------------------------------------------- 5. repositories
echo [5/8] Repositories
if exist "%REPO%\.git" (
    echo   kimodo present, kept as it is.
) else (
    git clone -b local https://github.com/alivnal/kimodo.git "%REPO%"
    if errorlevel 1 call :fail "git clone of alivnal/kimodo failed." & exit /b 1
)
git -C "%REPO%" rev-parse --verify --quiet refs/heads/local >nul
if errorlevel 1 (
    echo   WARNING: kimodo has no "local" branch. It lacks the NVIDIA v1.2 sync, the Mixamo
    echo            export and the fixes: git -C kimodo checkout local (alivnal/kimodo).
)
if exist "%REPO%\kimodo-viser\.git" (
    echo   kimodo-viser present, kept as it is.
) else (
    git clone https://github.com/nv-tlabs/kimodo-viser.git "%REPO%\kimodo-viser"
    if errorlevel 1 call :fail "git clone of kimodo-viser failed." & exit /b 1
)
echo.

:: ---------------------------------------------------------------- 6. kimodo, viser, motion_correction
echo [6/8] kimodo, viser and motion_correction into the venv
"%PY%" -m pip show -q kimodo >nul 2>&1
if errorlevel 1 (
    set SKIP_MOTION_CORRECTION_IN_SETUP=1
    "%PY%" -m pip install --disable-pip-version-check -q --no-deps -e "%REPO%"
    if errorlevel 1 call :fail "pip install -e kimodo failed." & exit /b 1
    echo   kimodo installed.
) else (
    echo   kimodo installed already.
)
"%PY%" -m pip show -q viser >nul 2>&1
if errorlevel 1 (
    "%PY%" -m pip install --disable-pip-version-check -q --no-deps -e "%REPO%\kimodo-viser"
    if errorlevel 1 call :fail "pip install -e kimodo-viser failed." & exit /b 1
    echo   viser installed.
) else (
    echo   viser installed already.
)
"%PY%" -m pip show -q motion_correction >nul 2>&1
if errorlevel 1 (
    if exist "%MC_WHEEL%" (
        "%PY%" -m pip install --disable-pip-version-check -q --no-deps "%MC_WHEEL%"
    ) else (
        "%PY%" -m pip install --disable-pip-version-check -q --no-deps "%MC_URL%"
    )
    if errorlevel 1 call :fail "motion_correction wheel could not be installed." & exit /b 1
    echo   motion_correction installed.
) else (
    echo   motion_correction installed already.
)
echo.

:: ---------------------------------------------------------------- 7. viser web client
echo [7/8] Viser web client
if exist "%VISER_CLIENT%\build\index.html" (
    echo   built already.
    goto :verify
)
set "PATH=C:\Program Files\nodejs;%PATH%"
where npm >nul 2>&1
if errorlevel 1 (
    echo   Node.js not found, installing Node.js LTS with winget...
    winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements
    set "PATH=C:\Program Files\nodejs;%PATH%"
)
where npm >nul 2>&1
if errorlevel 1 call :fail "npm still not found. Open a new terminal and run this installer again." & exit /b 1
pushd "%VISER_CLIENT%"
if exist .nodeenv rmdir /s /q .nodeenv
call npm install --legacy-peer-deps
if errorlevel 1 (popd & call :fail "npm install failed in the viser client." & exit /b 1)
call npx vite build
if errorlevel 1 (popd & call :fail "vite build failed in the viser client." & exit /b 1)
popd
echo   built.
echo.

:: ---------------------------------------------------------------- 8. check
:verify
echo [8/8] Checking the install
pushd "%ROOT%"
"%PY%" scripts\kimodo_check.py
set "CHECK=%errorlevel%"
popd
echo.
if not "%CHECK%"=="0" call :fail "The install check failed (see above)." & exit /b 1
echo ============================================
echo   Kimodo is ready. Start it with start_kimodo.bat
echo   (8 GB VRAM works as is; with less, start_kimodo.bat --offload)
echo ============================================
echo.
pause
exit /b 0

:fail
echo.
echo ERROR: %~1
echo.
pause
exit /b 1
