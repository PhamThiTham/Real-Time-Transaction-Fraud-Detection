# Real-time Fraud Detection helper files

Files:
- `start_realtime.bat`: opens the 6 pipeline stages in separate Windows terminals.
- `stop_realtime.bat`: stops the producer and Spark streaming jobs without deleting data.
- `Dockerfile`: custom Spark image definition.

Important:
1. Run the cleanup/reset procedure first when you want a completely fresh run.
2. Make sure all old Spark streams are stopped before deleting Kafka topics.
3. The start script assumes the existing paths and script names from the current project.
4. The Dockerfile assumes build-context folders `spark-apps/` and `models/`. If your current `docker-compose.yml` uses different folder names or bind mounts, keep the compose configuration as the source of truth or adjust these two COPY lines.

Nếu muốn chạy từng dòng lệnh thực hiện theo hướng dẫn bên dưới đây.

***6*** xóa dữ liệu để chạy lại từ đầu

cd /d D:\ThucTap_VinSmartFuture\run_realtime

Xóa checkpoint
docker exec -it spark rm -rf /opt/spark-data/checkpoint/lstm_inference
docker exec -it spark rm -rf /opt/spark-data/checkpoint/customer_history
docker exec -it spark rm -rf /opt/spark-data/checkpoint/feature_engineering
docker exec -it spark rm -rf /opt/spark-data/checkpoint/postgres_sink

Và xóa dữ liệu cũ:
docker exec -it spark rm -rf /opt/spark-data/features/*
docker exec -it spark rm -rf /opt/spark-data/history/*

Kiểm tra:
find /opt/spark-data -maxdepth 2 -type d | sort

Xóa dữ liệu PostgreSQL
docker exec -it postgres psql -U fraud -d fraud_detection -c "TRUNCATE TABLE fraud_predictions;"
Kiểm tra:
docker exec -it postgres psql -U fraud -d fraud_detection -c "SELECT COUNT(*) FROM fraud_predictions;"
Phải là: count = 0

Reset Kafka
Chỉ thực hiện bước này sau khi chắc chắn tất cả Spark stream đã dừng.
Xóa các topic:
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --delete --topic transactions --bootstrap-server kafka:29092
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --delete --topic transactions_features --bootstrap-server kafka:29092
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --delete --topic fraud_predictions --bootstrap-server kafka:29092

Kiểm tra đã xóa chưa:
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 --list

Tạo lại 3 topic sạch
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --create --topic transactions --bootstrap-server kafka:29092 --partitions 1 --replication-factor 1
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --create --topic transactions_features --bootstrap-server kafka:29092 --partitions 1 --replication-factor 1
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --create --topic fraud_predictions --bootstrap-server kafka:29092 --partitions 1 --replication-factor 1

Kiểm tra đã tạo lại topic chưa:
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 --list

Kiểm tra Kafka topic có thật sự rỗng:
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:29092 --topic transactions --from-beginning --timeout-ms 5000
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:29092 --topic transactions_features --from-beginning --timeout-ms 5000
docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server kafka:29092 --topic fraud_predictions --from-beginning --timeout-ms 5000

Kiểm tra cả 3 topic
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --describe --topic transactions --bootstrap-server kafka:29092
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --describe --topic transactions_features --bootstrap-server kafka:29092
docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --describe --topic fraud_predictions --bootstrap-server kafka:29092

***5*** chạy kafka postgres_sink

cd /d D:\ThucTap_VinSmartFuture\run_realtime

docker exec -it spark /opt/spark/bin/spark-submit --conf "spark.jars.ivy=/tmp/.ivy2" --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1,org.postgresql:postgresql:42.7.7 /opt/spark-apps/postgres_sink.py

***4*** chạy kafka lstm_inference

cd /d D:\ThucTap_VinSmartFuture\run_realtime

docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/lstm_inference.py

***3*** chạy kafka customer_history

cd /d D:\ThucTap_VinSmartFuture\run_realtime

docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2/cache --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/customer_history.py

***2*** chạy spark feature_engineering

cd /d D:\ThucTap_VinSmartFuture\run_realtime

docker exec -it spark /opt/spark/bin/spark-submit --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 /opt/spark-apps/feature_engineering.py

***1*** chạy kafka

cd /d D:\ThucTap_VinSmartFuture\run_realtime\producer

chạy run_1_banchmark thì sử dụng: python transaction_producer.py

chạy so sánh nhiều banchmark:
python transaction_producer.py --rate 10 --limit 500000 --print-every 1000
python transaction_producer.py --rate 50 --limit 500000 --print-every 1000
python transaction_producer.py --rate 100 --limit 500000 --print-every 1000
python transaction_producer.py --rate 500 --limit 500000 --print-every 1000


