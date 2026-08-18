import json
import random
import time

from datetime import datetime, timedelta, timezone

from kafka import KafkaProducer


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions"

# ------------------------------------------------------------
# SCENARIO
#
# 1 = Amount > 220 -> fraud
#
# 2 = 2 terminals/day compromised for 28 days
#
# 3 = 3 customers/day compromised for 14 days
# ------------------------------------------------------------

SCENARIO = 2


# ============================================================
# SIMULATION TIME
# ============================================================

START_DATETIME = datetime(
    2018,
    4,
    1,
    0,
    0,
    0
)

# Mỗi transaction thực tế cách nhau 2 giây
REAL_TIME_SLEEP = 2


# Mỗi transaction làm thời gian mô phỏng tăng 10 phút
SIMULATED_TIME_STEP = timedelta(
    minutes=10
)


# ============================================================
# FRAUD SIMULATION CONFIGURATION
# ============================================================

# Scenario 2:
#
# Bao nhiêu % transaction được gửi qua
# terminal compromised.
#
# 0.10 = 10%
#
# Nếu muốn fraud nhiều hơn:
#     0.20 = 20%
#
# Nếu muốn thực tế hơn:
#     0.05 = 5%
# ============================================================

COMPROMISED_TERMINAL_TRANSACTION_RATE = 0.10


# Scenario 3:
#
# Xác suất transaction của customer compromised
# trở thành fraud.
# ============================================================

COMPROMISED_CUSTOMER_FRAUD_RATE = 1 / 3


# ============================================================
# DATASET RANGE
# ============================================================

# ------------------------------------------------------------
# Customer IDs
#
# Hiện tại dùng 5 customer để dễ test.
#
# Khi chạy dữ liệu lớn có thể đổi thành:
#
# CUSTOMER_IDS = list(range(1, 5001))
# ------------------------------------------------------------

CUSTOMER_IDS = [
    1001,
    1002,
    1003,
    1004,
    1005
]


# ------------------------------------------------------------
# Terminal IDs
# ------------------------------------------------------------

TERMINAL_IDS = list(
    range(1, 5001)
)


# ============================================================
# KAFKA PRODUCER
# ============================================================

producer = KafkaProducer(

    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8")
)


# ============================================================
# GLOBAL STATE
# ============================================================

transaction_id = 0

current_datetime = START_DATETIME


# ============================================================
# COMPROMISED TERMINALS
#
# {
#     terminal_id: end_datetime
# }
# ============================================================

compromised_terminals = {}


# ============================================================
# COMPROMISED CUSTOMERS
#
# {
#     customer_id: end_datetime
# }
# ============================================================

compromised_customers = {}


# ============================================================
# LAST SIMULATED DAY
# ============================================================

last_simulated_day = None


# ============================================================
# CREATE DAILY COMPROMISED TERMINALS
#
# Scenario 2
#
# Mỗi ngày:
#     chọn ngẫu nhiên 2 terminal
#
# Terminal bị compromise trong 28 ngày.
# ============================================================

def create_daily_compromised_terminals(
    current_datetime
):

    terminals = random.sample(
        TERMINAL_IDS,
        2
    )

    end_datetime = (
        current_datetime
        + timedelta(days=28)
    )

    for terminal_id in terminals:

        compromised_terminals[
            terminal_id
        ] = end_datetime


    print()
    print("=" * 70)
    print("SCENARIO 2 - NEW COMPROMISED TERMINALS")
    print("=" * 70)

    for terminal_id in terminals:

        print(
            f"Terminal {terminal_id} | "
            f"start={current_datetime} | "
            f"end={end_datetime}"
        )

    print("=" * 70)
    print()


# ============================================================
# CREATE DAILY COMPROMISED CUSTOMERS
#
# Scenario 3
#
# Mỗi ngày:
#     chọn ngẫu nhiên 3 customers
#
# Customer bị compromise trong 14 ngày.
# ============================================================

def create_daily_compromised_customers(
    current_datetime
):

    # Tránh lỗi nếu số customer < 3
    sample_size = min(
        3,
        len(CUSTOMER_IDS)
    )

    customers = random.sample(
        CUSTOMER_IDS,
        sample_size
    )

    end_datetime = (
        current_datetime
        + timedelta(days=14)
    )

    for customer_id in customers:

        compromised_customers[
            customer_id
        ] = end_datetime


    print()
    print("=" * 70)
    print("SCENARIO 3 - NEW COMPROMISED CUSTOMERS")
    print("=" * 70)

    for customer_id in customers:

        print(
            f"Customer {customer_id} | "
            f"start={current_datetime} | "
            f"end={end_datetime}"
        )

    print("=" * 70)
    print()


# ============================================================
# CLEAN EXPIRED TERMINALS
# ============================================================

def clean_expired_terminals(
    current_datetime
):

    expired_terminals = []

    for terminal_id, end_datetime in (
        compromised_terminals.items()
    ):

        if current_datetime > end_datetime:

            expired_terminals.append(
                terminal_id
            )


    for terminal_id in expired_terminals:

        del compromised_terminals[
            terminal_id
        ]


# ============================================================
# CLEAN EXPIRED CUSTOMERS
# ============================================================

def clean_expired_customers(
    current_datetime
):

    expired_customers = []

    for customer_id, end_datetime in (
        compromised_customers.items()
    ):

        if current_datetime > end_datetime:

            expired_customers.append(
                customer_id
            )


    for customer_id in expired_customers:

        del compromised_customers[
            customer_id
        ]


# ============================================================
# GET ACTIVE COMPROMISED TERMINALS
# ============================================================

def get_active_compromised_terminals(
    current_datetime
):

    active_terminals = []

    for terminal_id, end_datetime in (
        compromised_terminals.items()
    ):

        if current_datetime <= end_datetime:

            active_terminals.append(
                terminal_id
            )

    return active_terminals


# ============================================================
# GET ACTIVE COMPROMISED CUSTOMERS
# ============================================================

def get_active_compromised_customers(
    current_datetime
):

    active_customers = []

    for customer_id, end_datetime in (
        compromised_customers.items()
    ):

        if current_datetime <= end_datetime:

            active_customers.append(
                customer_id
            )

    return active_customers


# ============================================================
# CHECK COMPROMISED TERMINAL
# ============================================================

def is_terminal_compromised(
    terminal_id,
    current_datetime
):

    if terminal_id not in compromised_terminals:

        return False

    end_datetime = (
        compromised_terminals[
            terminal_id
        ]
    )

    return current_datetime <= end_datetime


# ============================================================
# CHECK COMPROMISED CUSTOMER
# ============================================================

def is_customer_compromised(
    customer_id,
    current_datetime
):

    if customer_id not in compromised_customers:

        return False

    end_datetime = (
        compromised_customers[
            customer_id
        ]
    )

    return current_datetime <= end_datetime


# ============================================================
# GENERATE CUSTOMER
# ============================================================

def generate_customer():

    return random.choice(
        CUSTOMER_IDS
    )


# ============================================================
# GENERATE TERMINAL
# ============================================================

def generate_terminal(
    current_datetime
):

    # --------------------------------------------------------
    # Scenario 2
    #
    # Một tỷ lệ transaction được đưa vào
    # terminal compromised.
    # --------------------------------------------------------

    if SCENARIO == 2:

        if (
            random.random()
            < COMPROMISED_TERMINAL_TRANSACTION_RATE
        ):

            active_terminals = (
                get_active_compromised_terminals(
                    current_datetime
                )
            )

            if active_terminals:

                return random.choice(
                    active_terminals
                )


    # --------------------------------------------------------
    # Transaction bình thường
    # --------------------------------------------------------

    return random.choice(
        TERMINAL_IDS
    )


# ============================================================
# GENERATE AMOUNT
# ============================================================

def generate_amount():

    amount = round(
        random.lognormvariate(
            mu=5.5,
            sigma=1.0
        ),
        2
    )

    # Giới hạn amount
    amount = min(
        max(amount, 1),
        5000
    )

    return amount


# ============================================================
# GENERATE TRANSACTION
# ============================================================

def generate_transaction():

    global transaction_id
    global current_datetime
    global last_simulated_day


    # ========================================================
    # CHECK NEW SIMULATED DAY
    # ========================================================

    simulated_day = (
        current_datetime.date()
    )


    if (
        last_simulated_day is None
        or simulated_day != last_simulated_day
    ):

        # ----------------------------------------------------
        # Clean expired states
        # ----------------------------------------------------

        clean_expired_terminals(
            current_datetime
        )

        clean_expired_customers(
            current_datetime
        )


        # ----------------------------------------------------
        # Scenario 2
        # ----------------------------------------------------

        if SCENARIO == 2:

            create_daily_compromised_terminals(
                current_datetime
            )


        # ----------------------------------------------------
        # Scenario 3
        # ----------------------------------------------------

        if SCENARIO == 3:

            create_daily_compromised_customers(
                current_datetime
            )


        last_simulated_day = (
            simulated_day
        )


    # ========================================================
    # GENERATE CUSTOMER
    # ========================================================

    customer_id = generate_customer()


    # ========================================================
    # GENERATE TERMINAL
    # ========================================================

    terminal_id = generate_terminal(
        current_datetime
    )


    # ========================================================
    # GENERATE AMOUNT
    # ========================================================

    original_amount = generate_amount()

    amount = original_amount


    # ========================================================
    # DEFAULT
    # ========================================================

    fraud = 0

    scenario_status = "NORMAL"


    # ========================================================
    # SCENARIO 1
    #
    # Amount > 220 => fraud
    # ========================================================

    if SCENARIO == 1:

        if amount > 220:

            fraud = 1

            scenario_status = (
                "HIGH_AMOUNT_FRAUD"
            )


    # ========================================================
    # SCENARIO 2
    #
    # Compromised terminal => fraud
    # ========================================================

    elif SCENARIO == 2:

        if is_terminal_compromised(
            terminal_id,
            current_datetime
        ):

            fraud = 1

            scenario_status = (
                "COMPROMISED_TERMINAL"
            )


    # ========================================================
    # SCENARIO 3
    #
    # Compromised customer
    #
    # 1/3 transactions:
    #
    #     amount *= 5
    #     fraud = 1
    # ========================================================

    elif SCENARIO == 3:

        if is_customer_compromised(
            customer_id,
            current_datetime
        ):

            if (
                random.random()
                < COMPROMISED_CUSTOMER_FRAUD_RATE
            ):

                amount = round(
                    amount * 5,
                    2
                )

                amount = min(
                    amount,
                    5000
                )

                fraud = 1

                scenario_status = (
                    "COMPROMISED_CUSTOMER_FRAUD"
                )

            else:

                scenario_status = (
                    "COMPROMISED_CUSTOMER_NORMAL"
                )


    # ========================================================
    # TX_TIME_SECONDS
    # ========================================================

    tx_time_seconds = int(
        (
            current_datetime
            - START_DATETIME
        ).total_seconds()
    )


    # ========================================================
    # TX_TIME_DAYS
    # ========================================================

    tx_time_days = (
        tx_time_seconds
        // (24 * 60 * 60)
    )


    # ========================================================
    # PRODUCER TIMESTAMP
    #
    # Thời gian thực tại thời điểm transaction được tạo.
    # Dùng để đo END-TO-END LATENCY.
    #
    # Không dùng TX_DATETIME vì TX_DATETIME là thời gian
    # mô phỏng của dataset (2018).
    # ========================================================

    producer_timestamp = datetime.now(
        timezone.utc
    ).isoformat()


    # ========================================================
    # CREATE TRANSACTION
    # ========================================================

    transaction = {

        "TRANSACTION_ID":
            transaction_id,

        "PRODUCER_TIMESTAMP":
            producer_timestamp,

        "TX_DATETIME":
            current_datetime.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "CUSTOMER_ID":
            customer_id,

        "TERMINAL_ID":
            terminal_id,

        "TX_AMOUNT":
            amount,

        "TX_TIME_SECONDS":
            tx_time_seconds,

        "TX_TIME_DAYS":
            tx_time_days,

        "TX_FRAUD":
            fraud
    }


    # ========================================================
    # PRINT TRANSACTION
    # ========================================================

    print(
        f"Sent | "
        f"id={transaction_id} | "
        f"time={current_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"customer={customer_id} | "
        f"terminal={terminal_id} | "
        f"amount={original_amount:.2f} -> {amount:.2f} | "
        f"time_seconds={tx_time_seconds} | "
        f"time_days={tx_time_days} | "
        f"fraud={fraud} | "
        f"{scenario_status}"
    )


    # ========================================================
    # UPDATE STATE
    # ========================================================

    transaction_id += 1

    current_datetime += (
        SIMULATED_TIME_STEP
    )


    return transaction


# ============================================================
# START PRODUCER
# ============================================================

print("=" * 70)

print(
    "REAL-TIME FRAUD TRANSACTION PRODUCER"
)

print("=" * 70)

print(
    f"Kafka: "
    f"{KAFKA_BOOTSTRAP_SERVERS}"
)

print(
    f"Topic: "
    f"{KAFKA_TOPIC}"
)

print(
    f"Scenario: "
    f"{SCENARIO}"
)

if SCENARIO == 2:

    print(
        "Compromised terminal rate: "
        f"{COMPROMISED_TERMINAL_TRANSACTION_RATE * 100:.0f}%"
    )

elif SCENARIO == 3:

    print(
        "Compromised customer fraud rate: "
        f"{COMPROMISED_CUSTOMER_FRAUD_RATE * 100:.1f}%"
    )

print(
    f"Real-time sleep: "
    f"{REAL_TIME_SLEEP} seconds"
)

print(
    f"Simulated time step: "
    f"{SIMULATED_TIME_STEP}"
)

print(
    "Press Ctrl+C to stop."
)

print("=" * 70)


# ============================================================
# STREAMING LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Generate transaction
        # ----------------------------------------------------

        transaction = (
            generate_transaction()
        )


        # ----------------------------------------------------
        # Send transaction to Kafka
        # ----------------------------------------------------

        future = producer.send(
            KAFKA_TOPIC,
            value=transaction
        )


        # ----------------------------------------------------
        # Wait for Kafka acknowledgement
        # ----------------------------------------------------

        metadata = future.get(
            timeout=10
        )


        # ----------------------------------------------------
        # Kafka metadata
        # ----------------------------------------------------

        print(
            f"   Kafka -> "
            f"partition={metadata.partition}, "
            f"offset={metadata.offset}"
        )


        # ----------------------------------------------------
        # Real-time delay
        # ----------------------------------------------------

        time.sleep(
            REAL_TIME_SLEEP
        )


except KeyboardInterrupt:

    print(
        "\nStopping producer..."
    )


except Exception as e:

    print(
        "\nProducer error:"
    )

    print(e)


finally:

    print(
        "Flushing Kafka producer..."
    )

    producer.flush()

    producer.close()

    print(
        "Producer stopped."
    )