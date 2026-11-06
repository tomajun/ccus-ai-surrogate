@echo off
rem CCUS local pipeline launcher.
rem Pure ASCII on purpose (no Chinese) to avoid GBK garbling in cmd.exe.
rem Chinese logs are printed by the .py itself.
cd /d C:\workplace\ccus-ai-surrogate
if not exist .venv\Scripts\activate.bat (
  echo ERROR: .venv not found. Run tools\setup_env.bat first to create it.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python tools\run_local_pipeline.py
pause
