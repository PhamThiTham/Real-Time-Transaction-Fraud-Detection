@echo off
setlocal

title Real-Time Fraud Detection - STOP

cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================
echo     REAL-TIME FRAUD DETECTION - STOP
echo ============================================
echo.
echo This will stop ALL realtime pipeline windows:
echo.
echo   01 - Evaluate CPU/RAM
echo   02 - Kafka Producer
echo   03 - Feature Engineering
echo   04 - Customer History
echo   05 - LSTM Inference
echo   06 - PostgreSQL Sink
echo.
echo PostgreSQL data, Kafka topics and checkpoints
echo will NOT be deleted.
echo.

choice /C YN /M "Continue stopping the realtime pipeline"

if errorlevel 2 goto CANCEL

echo.
echo ============================================
echo Stopping realtime pipeline...
echo ============================================
echo.

echo [1/6] Stopping Evaluate CPU/RAM...
taskkill /FI "WINDOWTITLE eq 01 - Evaluate CPU/RAM*" /T /F >nul 2>&1

echo [2/6] Stopping Kafka Producer...
taskkill /FI "WINDOWTITLE eq 02 - Kafka Producer*" /T /F >nul 2>&1

echo [3/6] Stopping Feature Engineering...
taskkill /FI "WINDOWTITLE eq 03 - Feature Engineering*" /T /F >nul 2>&1

echo [4/6] Stopping Customer History...
taskkill /FI "WINDOWTITLE eq 04 - Customer History*" /T /F >nul 2>&1

echo [5/6] Stopping LSTM Inference...
taskkill /FI "WINDOWTITLE eq 05 - LSTM Inference*" /T /F >nul 2>&1

echo [6/6] Stopping PostgreSQL Sink...
taskkill /FI "WINDOWTITLE eq 06 - PostgreSQL Sink*" /T /F >nul 2>&1

echo.
echo ============================================
echo Stopping Spark jobs inside container...
echo ============================================

docker exec spark bash -lc "pkill -f 'postgres_sink.py' || true; pkill -f 'lstm_inference.py' || true; pkill -f 'customer_history.py' || true; pkill -f 'feature_engineering.py' || true"

echo.
echo ============================================
echo REALTIME PIPELINE STOPPED
echo ============================================
echo.
echo All 6 realtime CMD windows have been closed.
echo.
echo Docker containers remain running.
echo PostgreSQL data was NOT deleted.
echo Kafka data was NOT deleted.
echo Checkpoints were NOT deleted.
echo ============================================

timeout /t 2 /nobreak >nul

goto END

:CANCEL

echo.
echo ============================================
echo STOP CANCELLED
echo ============================================
echo.
echo No realtime windows were closed.
echo.

pause

:END
endlocal