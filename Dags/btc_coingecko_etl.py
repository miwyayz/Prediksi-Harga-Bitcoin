
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import mysql.connector
import requests

from airflow.decorators import dag, task


# ==================================================
# KONFIGURASI
# ==================================================

API_BASE_URL = "https://api.coingecko.com/api/v3"
COIN_ID = "bitcoin"
CURRENCY = "usd"
HISTORICAL_DAYS = 365


# ==================================================
# KONEKSI MYSQL
# ==================================================

def mysql_connection():
    """Membuat koneksi ke MySQL."""
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3307")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        connection_timeout=20,
    )


# ==================================================
# REQUEST COINGECKO API
# ==================================================

def coingecko_get(endpoint, params=None):
    """Mengirim request ke CoinGecko Demo API."""
    api_key = os.getenv("COINGECKO_DEMO_API_KEY")

    if not api_key:
        raise ValueError(
            "COINGECKO_DEMO_API_KEY belum diatur."
        )

    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        params=params or {},
        headers={
            "x-cg-demo-api-key": api_key,
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.json()


# ==================================================
# TRANSFORM HELPER
# ==================================================

def daily_rows(series):
    """
    Mengubah timestamp milidetik menjadi tanggal UTC.
    Jika terdapat beberapa nilai pada tanggal yang sama,
    nilai dari timestamp terbaru akan dipakai.
    """
    by_date = {}

    for timestamp_ms, value in series:
        if value is None:
            continue

        recorded_date = datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc,
        ).date().isoformat()

        timestamp_ms = int(timestamp_ms)

        if (
            recorded_date not in by_date
            or timestamp_ms > by_date[recorded_date][0]
        ):
            by_date[recorded_date] = (
                timestamp_ms,
                float(value),
            )

    return [
        {
            "recorded_date": date,
            "value": value,
        }
        for date, (_, value) in sorted(by_date.items())
    ]


def to_decimal(value):
    """Mengubah nilai API menjadi Decimal untuk MySQL."""
    if value is None:
        return None

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Nilai numerik tidak valid: {value}")

    if not result.is_finite():
        raise ValueError(f"Nilai numerik tidak valid: {value}")

    return result


# ==================================================
# DEFINISI DAG
# ==================================================

@dag(
    dag_id="btc_coingecko_etl",
    description=(
        "ETL harga dan market cap historis Bitcoin "
        "dari CoinGecko ke MySQL"
    ),
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "data-engineering",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=[
        "coingecko",
        "bitcoin",
        "mysql",
        "etl",
    ],
)
def btc_coingecko_etl():

    # ==================================================
    # 1. EXTRACT
    # ==================================================

    @task
    def extract():
        """
        Mengambil metadata Bitcoin serta riwayat
        harga dan market cap dari CoinGecko.
        """
        metadata = coingecko_get(
            "/coins/markets",
            {
                "vs_currency": CURRENCY,
                "ids": COIN_ID,
                "sparkline": "false",
            },
        )

        if not metadata:
            raise ValueError(
                "Data metadata Bitcoin tidak ditemukan."
            )

        coin = metadata[0]

        market_chart = coingecko_get(
            f"/coins/{COIN_ID}/market_chart",
            {
                "vs_currency": CURRENCY,
                "days": HISTORICAL_DAYS,
            },
        )

        if not market_chart.get("prices"):
            raise ValueError(
                "Data harga historis Bitcoin kosong."
            )

        return {
            "metadata": {
                "id": coin["id"],
                "symbol": coin["symbol"],
                "name": coin["name"],
            },
            "market_chart": market_chart,
        }

    # ==================================================
    # 2. TRANSFORM
    # ==================================================

    @task
    def transform(raw):
        """
        Membersihkan dan menyiapkan data agar siap
        disimpan ke tabel price dan market_cap.
        """
        metadata = raw["metadata"]
        chart = raw["market_chart"]

        prices = daily_rows(
            chart.get("prices", [])
        )

        market_caps = daily_rows(
            chart.get("market_caps", [])
        )

        if not prices:
            raise ValueError(
                "Tidak ada harga historis yang valid."
            )

        return {
            "coingecko_id": metadata["id"],
            "symbol": metadata["symbol"].lower(),
            "name": metadata["name"],
            "prices": prices,
            "market_caps": market_caps,
        }

    # ==================================================
    # 3. LOAD
    # ==================================================

    @task
    def load(data):
        """
        Membuat tabel jika belum tersedia, lalu
        menyimpan atau memperbarui data dengan UPSERT.
        """
        conn = mysql_connection()
        cursor = conn.cursor()

        statements = [
            """
            CREATE TABLE IF NOT EXISTS crypto (
                crypto_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                coingecko_id VARCHAR(100) NOT NULL UNIQUE,
                symbol VARCHAR(30) NOT NULL,
                name VARCHAR(150) NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS price (
                price_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                crypto_id BIGINT NOT NULL,
                recorded_date DATE NOT NULL,
                price_usd DECIMAL(24, 8) NOT NULL,
                CONSTRAINT fk_price_crypto
                    FOREIGN KEY (crypto_id)
                    REFERENCES crypto(crypto_id),
                CONSTRAINT uq_price_date
                    UNIQUE (crypto_id, recorded_date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_cap (
                market_cap_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                crypto_id BIGINT NOT NULL,
                recorded_date DATE NOT NULL,
                market_cap_usd DECIMAL(30, 2),
                CONSTRAINT fk_market_cap_crypto
                    FOREIGN KEY (crypto_id)
                    REFERENCES crypto(crypto_id),
                CONSTRAINT uq_market_cap_date
                    UNIQUE (crypto_id, recorded_date)
            )
            """,
        ]

        try:
            # A. CREATE SCHEMA
            for statement in statements:
                cursor.execute(statement)

            # B. UPSERT CRYPTO
            cursor.execute(
                """
                INSERT INTO crypto (
                    coingecko_id, symbol, name
                )
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    symbol = VALUES(symbol),
                    name = VALUES(name)
                """,
                (
                    data["coingecko_id"],
                    data["symbol"],
                    data["name"],
                ),
            )

            cursor.execute(
                """
                SELECT crypto_id
                FROM crypto
                WHERE coingecko_id = %s
                """,
                (data["coingecko_id"],),
            )

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    "crypto_id Bitcoin tidak ditemukan."
                )

            crypto_id = row[0]

            # C. UPSERT PRICE
            price_rows = [
                (
                    crypto_id,
                    row["recorded_date"],
                    to_decimal(row["value"]),
                )
                for row in data["prices"]
            ]

            if price_rows:
                cursor.executemany(
                    """
                    INSERT INTO price (
                        crypto_id,
                        recorded_date,
                        price_usd
                    )
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        price_usd = VALUES(price_usd)
                    """,
                    price_rows,
                )

            # D. UPSERT MARKET CAP
            market_cap_rows = [
                (
                    crypto_id,
                    row["recorded_date"],
                    to_decimal(row["value"]),
                )
                for row in data["market_caps"]
            ]

            if market_cap_rows:
                cursor.executemany(
                    """
                    INSERT INTO market_cap (
                        crypto_id,
                        recorded_date,
                        market_cap_usd
                    )
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        market_cap_usd = VALUES(market_cap_usd)
                    """,
                    market_cap_rows,
                )

            conn.commit()

            return {
                "crypto": data["coingecko_id"],
                "crypto_id": crypto_id,
                "prices_processed": len(price_rows),
                "market_caps_processed": len(
                    market_cap_rows
                ),
                "status": "success",
            }

        except Exception:
            conn.rollback()
            raise

        finally:
            cursor.close()
            conn.close()

    # ==================================================
    # DEPENDENSI TASK
    # ==================================================

    extracted = extract()
    transformed = transform(extracted)
    loaded = load(transformed)

    extracted >> transformed >> loaded


btc_coingecko_etl()
