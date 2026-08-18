import subprocess
import time
import csv
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Docker CPU/RAM during a real-time fraud test."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Monitoring duration in seconds. Default: 300."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds. Default: 1."
    )
    parser.add_argument(
        "--output",
        default="resource_metrics.csv",
        help="CSV output filename."
    )
    args = parser.parse_args()

    print("=" * 78)
    print("DOCKER RESOURCE MONITOR")
    print("=" * 78)
    print(f"Duration : {args.duration} seconds")
    print(f"Interval : {args.interval} seconds")
    print(f"Output   : {args.output}")
    print()

    rows = []

    start = time.monotonic()

    while time.monotonic() - start < args.duration:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}"
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            timestamp = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

            for line in result.stdout.splitlines():
                parts = line.split("|")

                if len(parts) != 4:
                    continue

                container, cpu, memory, memory_percent = parts

                rows.append([
                    timestamp,
                    container,
                    cpu,
                    memory,
                    memory_percent,
                ])

                print(
                    f"{timestamp} | "
                    f"{container:<15} | "
                    f"CPU {cpu:<8} | "
                    f"MEM {memory:<25} | "
                    f"{memory_percent}"
                )

        except Exception as exc:
            print(f"Monitoring error: {exc}")

        time.sleep(args.interval)

    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "container",
            "cpu_percent",
            "memory_usage",
            "memory_percent",
        ])

        writer.writerows(rows)

    print()
    print("=" * 78)
    print("RESOURCE MONITOR FINISHED")
    print("=" * 78)
    print(f"Samples saved: {len(rows)}")
    print(f"File: {args.output}")


if __name__ == "__main__":
    main()
