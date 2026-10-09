# Analisis Data Cryptocurrency dan Prediksi Harga Bitcoin Menggunakan Algoritma XGBoost

Proyek Data Science yang menganalisis data historis Bitcoin dan memprediksi harga penutupan satu hari berikutnya menggunakan XGBoost. Data diambil dari database MySQL `crypto_db`, yaitu hasil proses ETL Apache Airflow pada proyek `btc-coingecko`.

## Pertanyaan Bisnis

1. Bagaimana perkembangan harga Bitcoin, market capitalization, volume perdagangan, dan circulating supply berdasarkan data historis dalam database?
2. Bagaimana hubungan antara keempat variabel tersebut, dan apakah terdapat korelasi yang kuat?
3. Bagaimana karakteristik volatilitas dan perubahan harga Bitcoin, serta apakah perubahan volume dan market capitalization berkaitan dengan perubahan harga pada periode berikutnya?
4. Seberapa akurat XGBoost dalam memprediksi harga penutupan Bitcoin satu hari berikutnya?

## Sumber Data

Database MySQL `crypto_db` dengan lima tabel:

| Tabel | Kolom |
|---|---|
| `crypto` | `crypto_id`, `coingecko_id`, `symbol`, `name` |
| `price` | `price_id`, `crypto_id`, `recorded_date`, `price_usd` |
| `market_cap` | `market_cap_id`, `crypto_id`, `recorded_date`, `market_cap_usd` |
| `trading_volume` | `volume_id`, `crypto_id`, `recorded_date`, `volume_usd` |
| `supply` | `supply_id`, `crypto_id`, `recorded_date`, `circulating_supply` |

## Struktur Proyek

```text
btc-coingecko/
├── Dags/
│   └── btc_coingecko_etl.py        # DAG Airflow (proses ETL)
├── Notebook/                       # notebook analisis dan pemodelan
│   └── btc_xgboost_crypto.ipynb  
└── README.md
```

## Prasyarat

* Python 3.10 atau lebih baru
* MySQL berjalan di `127.0.0.1` port `3307` dengan database `crypto_db` yang sudah terisi oleh DAG Airflow
* Jupyter Notebook atau JupyterLab

Package yang digunakan:

```text
pandas, numpy, matplotlib, seaborn, scipy,
scikit-learn, xgboost, mysql-connector-python, ipython
```

## Instalasi

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost mysql-connector-python notebook
```

Dua sel pertama pada notebook juga menjalankan `pip install` untuk `xgboost`, `mysql-connector-python`, dan `scipy`.

## Konfigurasi

### Mode database (default)

Password database tidak ditulis di dalam notebook. Notebook membaca password dari environment variable `MYSQL_PASSWORD`. Jika variabel tersebut tidak ditemukan, notebook meminta password melalui input tersembunyi.

Mengatur environment variable di Windows (PowerShell):

```powershell
$env:MYSQL_PASSWORD = "password_anda"
jupyter notebook
```

Atau permanen (perlu membuka terminal baru setelahnya):

```powershell
setx MYSQL_PASSWORD "password_anda"
```

Konfigurasi koneksi berada pada sel "Konfigurasi Sumber Data":

```python
SUMBER_DATA = "database"
DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_NAME = "crypto_db"
DB_USER = "root"
```

## Cara Menjalankan

1. Pastikan DAG Airflow sudah berjalan dan tabel di `crypto_db` sudah terisi.
2. Buka `btc_xgboost_crypto.ipynb` melalui Jupyter.
3. Atur `SUMBER_DATA` dan konfigurasi koneksi bila diperlukan.
4. Jalankan seluruh sel dari atas ke bawah (`Kernel > Restart & Run All`).

## Alur Notebook

1. Judul, tujuan, dan pertanyaan bisnis
2. Import package (satu sel)
3. Konfigurasi dan koneksi database, termasuk validasi koneksi dan keberadaan tabel
4. Gathering data: lima tabel dibaca terpisah menjadi `df_crypto`, `df_price`, `df_market_cap`, `df_trading_volume`, `df_supply`
5. Assessing data sebelum cleaning
6. Cleaning data (hasil disimpan pada DataFrame berakhiran `_clean`)
7. Assessing data setelah cleaning
8. Integrasi tabel menjadi `df_btc` (`recorded_date`, `btc_price`, `market_cap_usd`, `volume_usd`, `circulating_supply`)
9. EDA untuk empat pertanyaan bisnis
10. Feature engineering untuk XGBoost
11. Pembagian data time series (80% training, 20% testing)
12. Modeling: baseline, tuning dengan `TimeSeriesSplit`, dan training XGBoost
13. Evaluasi: MAE, RMSE, R², MAPE, akurasi arah, grafik aktual vs prediksi, residual, dan feature importance
14. Prediksi satu hari berikutnya
15. Kesimpulan yang disusun otomatis dari hasil perhitungan notebook

## Keputusan Metodologi

**Cleaning**
* `crypto_id` Bitcoin dicari dari `coingecko_id == 'bitcoin'` (cadangan: `symbol == 'btc'`), tidak diasumsikan bernilai tertentu.
* Nilai tidak valid (harga, market cap, dan supply <= 0; volume < 0) diganti `NaN`, tidak dihapus dan tidak dinolkan.
* Duplikat diselesaikan dengan mempertahankan baris dengan ID terbesar (hasil muat ulang terbaru dari ETL).
* Baris tanpa harga dihapus karena harga adalah dasar rentang tanggal dan target prediksi.

**Integrasi**
* Harga menjadi dasar rentang tanggal (left join).
* Nilai kosong pada market cap, volume, dan supply diisi dengan forward fill pada urutan tanggal yang benar. Baris awal tanpa pengamatan valid pertama dikeluarkan dan tidak diisi.

**Pemodelan**
* Target: `target_price_next_day = btc_price.shift(-1)`, hanya valid jika tanggal berikutnya tepat 1 hari setelahnya.
* Fitur: harga saat ini, lag harga (1, 3, 7, 14 hari), return historis, rolling mean dan standar deviasi (7 dan 30 hari), market cap, volume, circulating supply, serta lag market cap dan volume.
* Pembagian data berdasarkan urutan waktu tanpa pengacakan, dengan validasi bahwa tidak ada tanggal training yang lebih baru daripada tanggal testing.
* Tuning hyperparameter hanya memakai data training dengan `TimeSeriesSplit`.
* Baseline: harga besok sama dengan harga hari ini.
* Model pembanding: XGBoost dengan target selisih harga, karena model berbasis pohon tidak mengekstrapolasi tingkat harga.
* Prediksi satu hari berikutnya dibuat dari fitur pada tanggal terakhir, tanpa memakai nilai market cap, volume, atau supply masa depan.

## Keluaran

Seluruh angka hasil analisis (statistik, korelasi, metrik evaluasi, dan prediksi) dihitung langsung oleh notebook dari data pada database. Tidak ada data dummy dan tidak ada angka yang ditulis manual.

## Keterbatasan

* Data hanya mencakup satu aset dengan frekuensi harian dalam periode terbatas.
* Fitur tidak memuat faktor eksternal seperti sentimen, berita, regulasi, kondisi makroekonomi, atau data on-chain.
* Model berbasis pohon tidak mengekstrapolasi harga di luar rentang data training.
* Evaluasi memakai satu pembagian training-testing.
* Hasil prediksi merupakan estimasi model, bukan kepastian harga, dan bukan saran investasi.

## Pemecahan Masalah

| Masalah | Penyebab dan solusi |
|---|---|
| Koneksi MySQL gagal | Periksa server MySQL, port `3307`, nama database, username, dan password |
| Tabel tidak ditemukan | Jalankan DAG Airflow agar tabel terbentuk |
| `ModuleNotFoundError` | Jalankan ulang sel `pip install` atau pasang package secara manual |
| File CSV tidak ditemukan | Sesuaikan `CSV_DIR` dan `CSV_PATHS` |
| Bitcoin tidak ditemukan | Periksa isi tabel `crypto` dan DAG yang mengisinya |
| Data kurang dari 60 baris | Tunggu data bertambah, karena lag 14 hari dan rolling 30 hari membutuhkan riwayat yang cukup |
