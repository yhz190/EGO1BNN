@echo off
setlocal
pushd "%~dp0"
title EGO1 BNN board evidence (371 frames)
echo ============================================================
echo  Board evidence playback: 371 frames via UART 921600
echo  Keep this window in front and screen-record for ~20 s.
echo ============================================================
echo.
set "EVID=%~dp0..\..\sim\board_evidence_371"
if defined EGO1_EVIDENCE_DIR set "EVID=%EGO1_EVIDENCE_DIR%"
"..\.venv\Scripts\python.exe" board_smoke.py COM6 921600 371 --dir "%EVID%"
echo.
echo ============================================================
echo  Expected: RESULT: 371/371 frames match the model exactly
echo            throughput ~58 frames/s
echo ============================================================
popd
pause
