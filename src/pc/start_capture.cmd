@echo off
chcp 65001 >nul
title EGO1 BNN 现场采集工具
cd /d "%~dp0"
echo ============================================
echo  EGO1 BNN live capture
echo  camera preview + Otsu + 1-bit frame + model
echo --------------------------------------------
echo  keys: 1=rock 2=paper 3=scissors 4=junk
echo        SPACE=shot  A=auto20  B=bg20  S=LOCK-env  R=reload  Q=quit
echo  (ENV must stay OK; POSE may vary freely)
echo ============================================
echo.
"..\.venv\Scripts\python.exe" capture_live.py
echo.
echo [exit] code=%ERRORLEVEL%
pause
