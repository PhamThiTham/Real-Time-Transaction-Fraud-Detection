@echo off
setlocal
title Real-Time Fraud Detection - STOP
cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================
echo   REAL-TIME FRAUD DETECTION - STOP
echo ============================================
echo.
echo WARNING: This stops the producer and Spark streaming jobs.
echo It does NOT delete PostgreSQL data, Kafka topics, checkpoints,
echo features, or history.
echo.

choice /C YN /M "Continue stopping the realtime pipeline"
if errorlevel 2 goto CANCEL

echo.
echo [1/2] Stopping Python transaction producer...
taskkill /FI "WINDOWTITLE eq 01 - Kafka Producer*" /T /F >nul 2>&1
taskkill /IM python.exe /FI "WINDOWTITLE eq 01 - Kafka Producer*" /T /F >nul 2>&1

echo [2/2] Stopping Spark streaming jobs...
docker exec spark bash -lc "pkill -f 'postgres_sink.py' || true; pkill -f 'lstm_inference.py' || true; pkill -f 'customer_history.py' || true; pkill -f 'feature_engineering.py' || true"
docker exec spark bash -lc "pkill -f 'spark-submit' || true"

echo.
echo ============================================
echo Pipeline stopped.
echo Docker containers remain running.
echo ============================================
pause
goto END

:CANCEL
echo.
echo Cancelled.
pause

:END
endlocal
