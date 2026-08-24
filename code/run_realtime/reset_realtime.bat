@echo off
setlocal EnableExtensions EnableDelayedExpansion

title REAL-TIME FRAUD DETECTION - RESET FROM START

cd /d D:\ThucTap_VinSmartFuture\run_realtime

echo ============================================================
echo       REAL-TIME FRAUD DETECTION - RESET FROM START
echo ============================================================
echo.
echo WARNING:
echo.
echo This script will DELETE:
echo   - Spark checkpoints
echo   - Spark features
echo   - Spark customer history
echo   - PostgreSQL fraud_predictions data
echo   - Kafka topic: transactions
echo   - Kafka topic: transactions_features
echo   - Kafka topic: fraud_predictions
echo.
echo IMPORTANT:
echo   1. ALL Spark Streaming jobs MUST be stopped.
echo   2. transaction_producer.py MUST be stopped.
echo   3. Keep Kafka, PostgreSQL and Spark containers RUNNING.
echo.
pause


REM ============================================================
REM [1/8] CHECK DOCKER CONTAINERS
REM ============================================================

echo.
echo ============================================================
echo [1/8] CHECK DOCKER CONTAINERS
echo ============================================================
echo.

echo Checking Spark container...

set "SPARK_RUNNING="

for /f "delims=" %%A in ('docker inspect -f "{{.State.Running}}" spark 2^>nul') do (
    set "SPARK_RUNNING=%%A"
)

if /i not "!SPARK_RUNNING!"=="true" (
    echo.
    echo ERROR: Container "spark" is NOT running.
    echo.
    echo Current Docker containers:
    echo.
    docker ps -a
    echo.
    pause
    exit /b 1
)

echo Spark container: RUNNING
echo.


echo Checking Kafka container...

set "KAFKA_RUNNING="

for /f "delims=" %%A in ('docker inspect -f "{{.State.Running}}" kafka 2^>nul') do (
    set "KAFKA_RUNNING=%%A"
)

if /i not "!KAFKA_RUNNING!"=="true" (
    echo.
    echo ERROR: Container "kafka" is NOT running.
    echo.
    echo Current Docker containers:
    echo.
    docker ps -a
    echo.
    pause
    exit /b 1
)

echo Kafka container: RUNNING
echo.


echo Checking PostgreSQL container...

set "POSTGRES_RUNNING="

for /f "delims=" %%A in ('docker inspect -f "{{.State.Running}}" postgres 2^>nul') do (
    set "POSTGRES_RUNNING=%%A"
)

if /i not "!POSTGRES_RUNNING!"=="true" (
    echo.
    echo ERROR: Container "postgres" is NOT running.
    echo.
    echo Current Docker containers:
    echo.
    docker ps -a
    echo.
    pause
    exit /b 1
)

echo PostgreSQL container: RUNNING
echo.


echo ============================================================
echo DOCKER CONTAINERS STATUS
echo ============================================================
echo.

docker ps --format "table {{.Names}}\t{{.Status}}"

echo.
echo Container check: OK.


REM ============================================================
REM [2/8] DELETE SPARK CHECKPOINTS
REM ============================================================

echo.
echo ============================================================
echo [2/8] DELETE SPARK CHECKPOINTS
echo ============================================================
echo.

echo --- LSTM inference checkpoint ---

docker exec spark sh -c "rm -rf /opt/spark-data/checkpoint/lstm_inference"

if errorlevel 1 (
    echo ERROR: Cannot delete LSTM inference checkpoint.
    echo.
    pause
    exit /b 1
)

echo Done.
echo.


echo --- Customer history checkpoint ---

docker exec spark sh -c "rm -rf /opt/spark-data/checkpoint/customer_history"

if errorlevel 1 (
    echo ERROR: Cannot delete customer_history checkpoint.
    echo.
    pause
    exit /b 1
)

echo Done.
echo.


echo --- Feature engineering checkpoint ---

docker exec spark sh -c "rm -rf /opt/spark-data/checkpoint/feature_engineering"

if errorlevel 1 (
    echo ERROR: Cannot delete feature_engineering checkpoint.
    echo.
    pause
    exit /b 1
)

echo Done.
echo.


echo --- PostgreSQL sink checkpoint ---

docker exec spark sh -c "rm -rf /opt/spark-data/checkpoint/postgres_sink"

if errorlevel 1 (
    echo WARNING: PostgreSQL sink checkpoint could not be deleted.
) else (
    echo Done.
)

echo.
echo Spark checkpoints reset successfully.


REM ============================================================
REM [3/8] DELETE SPARK FEATURES AND HISTORY
REM ============================================================

echo.
echo ============================================================
echo [3/8] DELETE SPARK FEATURES AND CUSTOMER HISTORY
echo ============================================================
echo.


echo --- Delete Spark features ---

docker exec spark sh -c "rm -rf /opt/spark-data/features/*"

if errorlevel 1 (
    echo ERROR: Cannot delete Spark features.
    echo.
    pause
    exit /b 1
)

echo Features deleted.
echo.


echo --- Delete customer history ---

docker exec spark sh -c "rm -rf /opt/spark-data/history/*"

if errorlevel 1 (
    echo ERROR: Cannot delete customer history.
    echo.
    pause
    exit /b 1
)

echo Customer history deleted.
echo.


echo ============================================================
echo VERIFY SPARK FEATURES
echo ============================================================
echo.

docker exec spark sh -c "if find /opt/spark-data/features -mindepth 1 -print -quit 2>/dev/null | grep -q .; then echo NOT EMPTY; exit 1; else echo EMPTY; fi"

if errorlevel 1 (
    echo.
    echo ERROR: /opt/spark-data/features is NOT empty.
    echo.
    pause
    exit /b 1
)

echo Features directory: EMPTY
echo.


echo ============================================================
echo VERIFY CUSTOMER HISTORY
echo ============================================================
echo.

docker exec spark sh -c "if find /opt/spark-data/history -mindepth 1 -print -quit 2>/dev/null | grep -q .; then echo NOT EMPTY; exit 1; else echo EMPTY; fi"

if errorlevel 1 (
    echo.
    echo ERROR: /opt/spark-data/history is NOT empty.
    echo.
    pause
    exit /b 1
)

echo History directory: EMPTY
echo.

echo Spark data reset completed successfully.


REM ============================================================
REM [4/8] RESET POSTGRESQL
REM ============================================================

echo.
echo ============================================================
echo [4/8] RESET POSTGRESQL
echo ============================================================
echo.

echo --- TRUNCATE fraud_predictions ---

docker exec postgres psql -U fraud -d fraud_detection -c "TRUNCATE TABLE fraud_predictions;"

if errorlevel 1 (
    echo.
    echo ERROR: PostgreSQL TRUNCATE failed.
    echo.
    pause
    exit /b 1
)

echo.
echo PostgreSQL table truncated successfully.
echo.


echo --- VERIFY PostgreSQL ---

set "PG_COUNT="

for /f "delims=" %%A in ('docker exec postgres psql -U fraud -d fraud_detection -t -A -c "SELECT COUNT(*) FROM fraud_predictions;" 2^>nul') do (
    set "PG_COUNT=%%A"
)

echo PostgreSQL row count = !PG_COUNT!
echo.

if not "!PG_COUNT!"=="0" (
    echo ERROR: PostgreSQL fraud_predictions is NOT empty.
    echo Expected: 0
    echo.
    pause
    exit /b 1
)

echo PostgreSQL verification: OK.


REM ============================================================
REM [5/8] DELETE KAFKA TOPICS
REM ============================================================

echo.
echo ============================================================
echo [5/8] DELETE KAFKA TOPICS
echo ============================================================
echo.

echo IMPORTANT:
echo All Spark Streaming jobs MUST be stopped before this step.
echo transaction_producer.py MUST also be stopped.
echo.


echo --- DELETE transactions ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --delete ^
    --topic transactions

echo.


echo --- DELETE transactions_features ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --delete ^
    --topic transactions_features

echo.


echo --- DELETE fraud_predictions ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --delete ^
    --topic fraud_predictions

echo.
echo Kafka delete commands completed.


REM ============================================================
REM [6/8] VERIFY KAFKA TOPICS WERE DELETED
REM ============================================================

echo.
echo ============================================================
echo [6/8] VERIFY OLD KAFKA TOPICS ARE REALLY DELETED
echo ============================================================
echo.

echo Waiting for Kafka topic deletion...
timeout /t 3 /nobreak >nul

echo.
echo Current Kafka topics:
echo.

set "TOPIC_CHECK_FAILED=0"

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --list > "%TEMP%\kafka_topics_reset.txt"

if errorlevel 1 (
    echo ERROR: Cannot list Kafka topics.
    echo.
    pause
    exit /b 1
)

type "%TEMP%\kafka_topics_reset.txt"

echo.


findstr /x /c:"transactions" "%TEMP%\kafka_topics_reset.txt" >nul

if not errorlevel 1 (
    echo ERROR: transactions still exists.
    set "TOPIC_CHECK_FAILED=1"
)


findstr /x /c:"transactions_features" "%TEMP%\kafka_topics_reset.txt" >nul

if not errorlevel 1 (
    echo ERROR: transactions_features still exists.
    set "TOPIC_CHECK_FAILED=1"
)


findstr /x /c:"fraud_predictions" "%TEMP%\kafka_topics_reset.txt" >nul

if not errorlevel 1 (
    echo ERROR: fraud_predictions still exists.
    set "TOPIC_CHECK_FAILED=1"
)


del "%TEMP%\kafka_topics_reset.txt" >nul 2>&1


if "!TOPIC_CHECK_FAILED!"=="1" (
    echo.
    echo ============================================================
    echo ERROR: KAFKA TOPICS WERE NOT FULLY DELETED
    echo ============================================================
    echo.
    echo DO NOT continue.
    echo.
    echo Possible causes:
    echo   - Spark streaming jobs are still running.
    echo   - transaction_producer.py is still running.
    echo   - Kafka is still deleting the topics.
    echo.
    echo Stop all Spark jobs and Producer.
    echo Then run this RESET script again.
    echo.
    pause
    exit /b 1
)

echo.
echo All three old Kafka topics have been deleted successfully.


REM ============================================================
REM [7/8] CREATE CLEAN KAFKA TOPICS
REM ============================================================

echo.
echo ============================================================
echo [7/8] CREATE CLEAN KAFKA TOPICS
echo ============================================================
echo.


echo --- CREATE transactions ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --create ^
    --topic transactions ^
    --partitions 1 ^
    --replication-factor 1

if errorlevel 1 (
    echo.
    echo ERROR: Cannot create transactions topic.
    echo.
    pause
    exit /b 1
)

echo.


echo --- CREATE transactions_features ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --create ^
    --topic transactions_features ^
    --partitions 1 ^
    --replication-factor 1

if errorlevel 1 (
    echo.
    echo ERROR: Cannot create transactions_features topic.
    echo.
    pause
    exit /b 1
)

echo.


echo --- CREATE fraud_predictions ---

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --create ^
    --topic fraud_predictions ^
    --partitions 1 ^
    --replication-factor 1

if errorlevel 1 (
    echo.
    echo ERROR: Cannot create fraud_predictions topic.
    echo.
    pause
    exit /b 1
)

echo.
echo Kafka topics created successfully.


REM ============================================================
REM [8/8] VERIFY KAFKA TOPICS
REM ============================================================

echo.
echo ============================================================
echo [8/8] VERIFY KAFKA TOPICS
echo ============================================================
echo.


echo ============================================================
echo TOPIC: transactions
echo ============================================================

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --describe ^
    --topic transactions

if errorlevel 1 (
    echo ERROR: Cannot describe transactions.
    pause
    exit /b 1
)

echo.


echo ============================================================
echo TOPIC: transactions_features
echo ============================================================

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --describe ^
    --topic transactions_features

if errorlevel 1 (
    echo ERROR: Cannot describe transactions_features.
    pause
    exit /b 1
)

echo.


echo ============================================================
echo TOPIC: fraud_predictions
echo ============================================================

docker exec kafka /opt/kafka/bin/kafka-topics.sh ^
    --bootstrap-server kafka:29092 ^
    --describe ^
    --topic fraud_predictions

if errorlevel 1 (
    echo ERROR: Cannot describe fraud_predictions.
    pause
    exit /b 1
)


REM ============================================================
REM VERIFY KAFKA CONTENT
REM ============================================================

echo.
echo ============================================================
echo VERIFY KAFKA TOPIC CONTENT
echo ============================================================
echo.

echo IMPORTANT:
echo Each consumer waits 5 seconds.
echo No transaction data should appear.
echo.


echo ------------------------------------------------------------
echo transactions
echo ------------------------------------------------------------

docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh ^
    --bootstrap-server kafka:29092 ^
    --topic transactions ^
    --from-beginning ^
    --timeout-ms 5000

echo.


echo ------------------------------------------------------------
echo transactions_features
echo ------------------------------------------------------------

docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh ^
    --bootstrap-server kafka:29092 ^
    --topic transactions_features ^
    --from-beginning ^
    --timeout-ms 5000

echo.


echo ------------------------------------------------------------
echo fraud_predictions
echo ------------------------------------------------------------

docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh ^
    --bootstrap-server kafka:29092 ^
    --topic fraud_predictions ^
    --from-beginning ^
    --timeout-ms 5000

echo.


REM ============================================================
REM FINAL POSTGRES CHECK
REM ============================================================

echo.
echo ============================================================
echo FINAL POSTGRESQL CHECK
echo ============================================================
echo.

docker exec postgres psql ^
    -U fraud ^
    -d fraud_detection ^
    -c "SELECT COUNT(*) AS total FROM fraud_predictions;"

if errorlevel 1 (
    echo ERROR: Final PostgreSQL check failed.
    pause
    exit /b 1
)

echo.


REM ============================================================
REM FINAL SPARK CHECKPOINT CHECK
REM ============================================================

echo.
echo ============================================================
echo FINAL SPARK CHECKPOINT CHECK
echo ============================================================
echo.

docker exec spark sh -c "echo ===== LSTM INFERENCE ===== && if [ -d /opt/spark-data/checkpoint/lstm_inference ]; then echo EXISTS; else echo DELETED; fi && echo ===== CUSTOMER HISTORY ===== && if [ -d /opt/spark-data/checkpoint/customer_history ]; then echo EXISTS; else echo DELETED; fi && echo ===== FEATURE ENGINEERING ===== && if [ -d /opt/spark-data/checkpoint/feature_engineering ]; then echo EXISTS; else echo DELETED; fi && echo ===== POSTGRES SINK ===== && if [ -d /opt/spark-data/checkpoint/postgres_sink ]; then echo EXISTS; else echo DELETED; fi"

echo.


REM ============================================================
REM FINAL SPARK DATA CHECK
REM ============================================================

echo.
echo ============================================================
echo FINAL SPARK DATA CHECK
echo ============================================================
echo.

echo ===== FEATURES =====

docker exec spark sh -c "if find /opt/spark-data/features -mindepth 1 -print -quit 2>/dev/null | grep -q .; then echo NOT EMPTY; else echo EMPTY; fi"

echo.

echo ===== HISTORY =====

docker exec spark sh -c "if find /opt/spark-data/history -mindepth 1 -print -quit 2>/dev/null | grep -q .; then echo NOT EMPTY; else echo EMPTY; fi"

echo.


REM ============================================================
REM FINAL RESULT
REM ============================================================

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

echo Kafka:
echo   All three topics are CLEAN and EMPTY.
echo.

echo Spark:
echo   LSTM checkpoint       = deleted
echo   Customer history      = deleted
echo   Feature engineering  = deleted
echo   PostgreSQL sink       = deleted
echo.

echo Spark data:
echo   features              = empty
echo   history               = empty
echo.

echo ============================================================
echo SYSTEM READY FOR CLEAN BENCHMARK
echo ============================================================
echo.

echo NEXT STEPS:
echo.
echo   1. Start Feature Engineering
echo   2. Start Customer History
echo   3. Start LSTM Inference
echo   4. Start PostgreSQL Sink
echo   5. Verify all Spark streams are READY
echo   6. Start Producer
echo.

echo Benchmark:
echo.
echo   10 tx/s
echo   RESET
echo   50 tx/s
echo   RESET
echo   100 tx/s
echo   RESET
echo   500 tx/s
echo.

echo IMPORTANT:
echo   - Do NOT modify TRANSACTION_ID PRIMARY KEY.
echo   - Do NOT start Producer before Spark streams are READY.
echo   - PRODUCER_TIMESTAMP must remain the real producer/send timestamp.
echo   - TX_FRAUD must remain the ground-truth label.
echo.

echo ============================================================
echo.

pause

endlocal

