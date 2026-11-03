@echo off
setlocal
pushd "%~dp0"
title EGO1 BNN live demo (recording mode)
echo ============================================================
echo  EGO1 BNN live demo - recording mode
echo  keys: q=quit  f=fullscreen  g=guide-box  h=hysteresis
echo  Keep your hand inside the yellow guide box.
echo ============================================================
"..\.venv\Scripts\python.exe" demo_live.py COM6 921600 --rec
popd
pause
