@echo off
setlocal EnableExtensions
title REAL-TIME FRAUD DETECTION - RESET FROM START

cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================================
echo       REAL-TIME FRAUD DETECTION - RESET FROM START
echo ============================================================
echo.
echo WARNING:
echo This script will DELETE:
echo   - Spark checkpoints
echo   - Spark features
echo   - Spark customer history
echo   - PostgreSQL fraud_predictions data
echo   - Kafka topics: transactions
echo   - Kafka topics: transactions_features
echo   - Kafka topics: fraud_predictions
echo.
echo Make sure ALL Spark streaming windows are STOPPED.
echo.
pause

echo.
echo ============================================================
echo [1/7] XOA CHECKPOINT - LSTM INFERENCE
echo ============================================================
docker exec spark rm -rf /tmp/checkpoint/lstm_inference

if errorlevel 1 (
    echo ERROR: Cannot delete LSTM checkpoint.
    pause
    exit /b 1
)

echo Done.

echo.
echo ============================================================
echo [2/7] XOA CHECKPOINT - CUSTOMER HISTORY
echo ============================================================
docker exec spark rm -rf /tmp/checkpoint/customer_history

if errorlevel 1 (
    echo ERROR: Cannot delete customer_history checkpoint.
    pause
    exit /b 1
)

echo Done.

echo.
echo ============================================================
echo [3/7] XOA CHECKPOINT + DU LIEU CU CUA SPARK
echo ============================================================

docker exec spark rm -rf /tmp/checkpoint/feature_engineering
docker exec spark rm -rf /opt/spark-data/checkpoint/postgres_sink

docker exec spark rm -rf /opt/spark-data/features/*
docker exec spark rm -rf /opt/spark-data/history/*

if errorlevel 1 (
    echo WARNING: Some Spark data may not have been deleted.
)

echo Spark data reset completed.

echo.
echo ============================================================
echo [4/7] XOA DU LIEU POSTGRESQL
echo ============================================================

docker exec postgres psql -U fraud -d fraud_detection -c "TRUNCATE TABLE fraud_predictions;"

if errorlevel 1 (
    echo ERROR: PostgreSQL TRUNCATE failed.
    pause
    exit /b 1
)

echo PostgreSQL table truncated.

echo.
echo ============================================================
echo [5/7] KIEM TRA POSTGRESQL
echo ============================================================

docker exec postgres psql -U fraud -d fraud_detection -c "SELECT COUNT(*) AS count FROM fraud_predictions;"

echo.
echo ============================================================
echo PHIA TREN PHAI HIEN THI:
echo count = 0
echo ============================================================
echo.
pause

echo.
echo ============================================================
echo [6/7] RESET KAFKA TOPICS
echo ============================================================
echo.

echo --- XOA TOPIC: transactions ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --delete --topic transactions --bootstrap-server kafka:29092

echo.
echo --- XOA TOPIC: transactions_features ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --delete --topic transactions_features --bootstrap-server kafka:29092

echo.
echo --- XOA TOPIC: fraud_predictions ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --delete --topic fraud_predictions --bootstrap-server kafka:29092

echo.
echo Kafka topics deleted.

echo.
echo ============================================================
echo TAO LAI 3 KAFKA TOPIC SACH
echo ============================================================

echo.
echo --- TAO transactions ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --topic transactions --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1

echo.
echo --- TAO transactions_features ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --topic transactions_features --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1

echo.
echo --- TAO fraud_predictions ---
docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --topic fraud_predictions --bootstrap-server kafka:29092 --partitions 3 --replication-factor 1

echo.
echo ============================================================
echo [7/7] KIEM TRA 3 KAFKA TOPIC
echo ============================================================

echo.
echo ============================================================
echo TOPIC: transactions
echo ============================================================
docker exec kafka /opt/kafka/bin/kafka-topics.sh --describe --topic transactions --bootstrap-server kafka:29092

echo.
echo ============================================================
echo TOPIC: transactions_features
echo ============================================================
docker exec kafka /opt/kafka/bin/kafka-topics.sh --describe --topic transactions_features --bootstrap-server kafka:29092

echo.
echo ============================================================
echo TOPIC: fraud_predictions
echo ============================================================
docker exec kafka /opt/kafka/bin/kafka-topics.sh --describe --topic fraud_predictions --bootstrap-server kafka:29092

echo.
echo ============================================================
echo              RESET HOAN TAT
echo ============================================================
echo.
echo PostgreSQL:
echo   fraud_predictions = 0 records
echo.
echo Kafka:
echo   transactions          = 3 partitions
echo   transactions_features = 3 partitions
echo   fraud_predictions     = 3 partitions
echo.
echo Spark checkpoints:
echo   Deleted
echo.
echo Spark features/history:
echo   Deleted
echo.
echo He thong da san sang de chay lai tu dau.
echo ============================================================
echo.
pause

endlocal

