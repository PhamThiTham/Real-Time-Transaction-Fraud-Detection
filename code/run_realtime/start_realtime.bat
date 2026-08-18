@echo off
setlocal
title Real-Time Fraud Detection - START
cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================
echo   REAL-TIME FRAUD DETECTION - START
echo ============================================
echo.

echo [1/5] Starting PostgreSQL sink...
start "05 - PostgreSQL Sink" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf "spark.jars.ivy=/tmp/.ivy2" --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.postgresql:postgresql:42.7.7 /opt/spark-apps/postgres_sink.py"
timeout /t 5 /nobreak >nul

echo [2/5] Starting LSTM inference...
start "04 - LSTM Inference" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/lstm_inference.py"
timeout /t 5 /nobreak >nul

echo [3/5] Starting customer history...
start "03 - Customer History" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/customer_history.py"
timeout /t 5 /nobreak >nul

echo [4/5] Starting feature engineering...
start "02 - Feature Engineering" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime && docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/feature_engineering.py"
timeout /t 5 /nobreak >nul

echo [5/5] Starting Kafka transaction producer...
start "01 - Kafka Producer" cmd /k "cd /d D:\ThucTap_VinSmartFuture\run_realtime\producer && python transaction_producer.py"

echo.
echo ============================================
echo All 5 processes have been opened.
echo Keep all windows running.
echo ============================================
pause
