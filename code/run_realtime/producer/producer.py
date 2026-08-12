import json
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions"


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)


def generate_transaction():
    customer_id = random.randint(1001, 1010)

    amount = round(random.uniform(10, 5000), 2)

    fraud = 1 if amount > 4000 else 0

    transaction = {
        "CUSTOMER_ID": customer_id,
        "TX_AMOUNT": amount,
        "TX_FRAUD": fraud,
        "TX_DATETIME": datetime.now(timezone.utc).isoformat(),
    }

    return transaction


print("Python Transaction Producer started...")
print(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
print(f"Topic: {KAFKA_TOPIC}")
print("Press Ctrl+C to stop.\n")


try:
    while True:
        transaction = generate_transaction()

        future = producer.send(
            KAFKA_TOPIC,
            value=transaction
        )

        metadata = future.get(timeout=10)

        print(
            f"Sent transaction | "
            f"customer={transaction['CUSTOMER_ID']} | "
            f"amount={transaction['TX_AMOUNT']} | "
            f"fraud={transaction['TX_FRAUD']} | "
            f"partition={metadata.partition} | "
            f"offset={metadata.offset}"
        )

        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping producer...")

finally:
    producer.flush()
    producer.close()
    print("Producer stopped.")