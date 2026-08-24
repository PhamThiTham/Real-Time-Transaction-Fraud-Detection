@echo off
setlocal
title Real-Time Fraud Detection - START
cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================
echo   REAL-TIME FRAUD DETECTION - START
echo ============================================
echo.

echo [1/6] Starting PostgreSQL sink...
start "06 - PostgreSQL Sink" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf "spark.jars.ivy=/tmp/.ivy2" --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.postgresql:postgresql:42.7.7 /opt/spark-apps/postgres_sink.py"
timeout /t 5 /nobreak >nul

echo [2/6] Starting LSTM inference...
start "05 - LSTM Inference" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/lstm_inference.py"
timeout /t 5 /nobreak >nul

echo [3/6] Starting customer history...
start "04 - Customer History" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/customer_history.py"
timeout /t 5 /nobreak >nul

echo [4/6] Starting feature engineering...
start "03 - Feature Engineering" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/feature_engineering.py"
timeout /t 5 /nobreak >nul

echo [5/6] Starting Kafka transaction producer...
start "02 - Kafka Producer" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime\producer && python transaction_producer.py --rate 100 --limit 500000 --print-every 1000"

echo [6/6] Starting Evaluate CPU/RAM...
start "01 - Evaluate CPU/RAM" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime\evaluate && python monitor_resources.py --duration 300 --interval 1"

echo.
echo ============================================
echo All 6 processes have been opened.
echo Keep all windows running.
echo ============================================
pause
