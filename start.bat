@echo off
REM One-command start for Windows (no Docker required).
REM Everything stays inside this project folder - nothing is written to C:.
setlocal
cd /d "%~dp0"

set PYTHON=python
where %PYTHON% >nul 2>nul || (echo [start] Python 3.10+ is required on PATH & exit /b 1)

REM keep every download cache inside the project folder (nothing on C:)
set "PIP_CACHE_DIR=%cd%\.runtime\pip-cache"

if not exist .venv\Scripts\python.exe (
    echo [start] creating virtualenv ^(.venv^)...
    %PYTHON% -m venv .venv || exit /b 1
)
set VPY=.venv\Scripts\python.exe

if not exist .venv\.deps-installed (
    echo [start] installing dependencies...
    %VPY% -m pip install --upgrade pip -q || exit /b 1
    %VPY% -m pip install -q -r requirements.txt || exit /b 1
    echo ok> .venv\.deps-installed
)

REM --- ensure the local LLM backend (Ollama, portable, project-local) ---------
set OLLAMA_BIN=.runtime\ollama\ollama.exe
set OLLAMA_MODELS=%cd%\.runtime\ollama-models
set OLLAMA_HOST=127.0.0.1:11434
set LLM_MODEL=%LLM_MODEL%
if "%LLM_MODEL%"=="" set LLM_MODEL=llama3.2:3b

if not exist "%OLLAMA_BIN%" (
    echo [start] downloading portable Ollama into .runtime^(~300 MB, one time^)...
    powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; New-Item -ItemType Directory -Force -Path '.runtime\ollama-dl' | Out-Null; Invoke-WebRequest -Uri 'https://ollama.com/download/ollama-windows-amd64.zip' -OutFile '.runtime\ollama-dl\ollama.zip'; Expand-Archive -Force -Path '.runtime\ollama-dl\ollama.zip' -DestinationPath '.runtime\ollama'" || goto :no_llm
)

curl -s -o nul http://127.0.0.1:11434/ 2>nul
if errorlevel 1 (
    echo [start] starting Ollama server...
    start "ollama" /min cmd /c ""%OLLAMA_BIN%" serve >"%cd%\.runtime\ollama-serve.log" 2>&1"
    timeout /t 4 /nobreak >nul
)

%OLLAMA_BIN% list 2>nul | findstr /c:"%LLM_MODEL%" >nul
if errorlevel 1 (
    echo [start] pulling %LLM_MODEL% ^(~2 GB, one time^)...
    %OLLAMA_BIN% pull %LLM_MODEL% || goto :no_llm
)
goto :llm_ready

:no_llm
echo [start] WARNING: could not prepare the local LLM. Starting with the
echo [start] deterministic extractive fallback ^(LLM_PROVIDER=extractive^).
echo [start] Answers work but are quote-based; see README to enable the real model.
set LLM_PROVIDER=extractive
goto :serve

:llm_ready
set LLM_PROVIDER=ollama

:serve
echo.
echo [start] Commute Policy Assistant on http://127.0.0.1:8000  ^(Ctrl+C to stop^)
echo [start] token:   %VPY% -m app.cli issue-token --company acme --role employee
echo [start] ask:     see README for curl examples
echo.
%VPY% -m uvicorn app.main:app --host 0.0.0.0 --port 8000
