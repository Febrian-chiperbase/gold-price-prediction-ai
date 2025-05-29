# Analisis dan Prediksi Harga Emas Menggunakan AI (LSTM & Prophet)

![Analisis Harga Emas Menggunakan LSTM](images/gold_prediction_lstm.png)

Repositori ini berisi kode Python untuk melakukan analisis dan prediksi harga emas menggunakan model *Deep Learning* (LSTM) dan model deret waktu Prophet. Proyek ini bertujuan untuk mengeksplorasi bagaimana teknik AI dapat diterapkan untuk memahami dan mencoba memprediksi pergerakan harga komoditas.

## Fitur Utama

* Pengambilan data harga emas historis secara otomatis dari Yahoo Finance.
* Pra-pemrosesan data termasuk normalisasi dan pembuatan sekuens untuk LSTM.
* Implementasi model LSTM menggunakan TensorFlow/Keras untuk prediksi harga.
* Implementasi model Prophet untuk peramalan deret waktu dengan analisis tren dan musiman.
* Evaluasi model menggunakan metrik RMSE (Root Mean Squared Error) dan MAE (Mean Absolute Error).
* Visualisasi hasil prediksi dibandingkan dengan harga aktual.
* Prediksi harga emas untuk beberapa hari ke depan.

## Teknologi yang Digunakan

* **Python 3.11+**
* **Pustaka Utama:**
    * `pandas`: Manipulasi dan analisis data.
    * `numpy`: Komputasi numerik.
    * `yfinance`: Mengunduh data pasar keuangan dari Yahoo Finance.
    * `scikit-learn`: Pra-pemrosesan data dan metrik evaluasi.
    * `tensorflow`: Membangun dan melatih model LSTM (Deep Learning).
    * `prophet`: Model peramalan deret waktu dari Meta (Facebook).
    * `matplotlib`: Membuat plot dan visualisasi data.

## Struktur Repositori

```
├── .gitignore          # File yang diabaikan oleh Git
├── README.md           # Dokumentasi utama ini
├── requirements.txt    # Daftar pustaka Python yang dibutuhkan
├── src/                # Direktori berisi kode sumber utama
│   └── analisis_emas.py # Skrip utama untuk analisis dan prediksi
└── images/             # Direktori berisi gambar hasil plot

## Instalasi dan Penggunaan

**Prasyarat:**

* Python 3.11 atau lebih baru.
* `pip` (Python package installer).
* `venv` (untuk lingkungan virtual, sangat direkomendasikan).
* Git.

**Langkah-langkah:**

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/Febrian-chiperbase/gold-price-prediction-ai.git](https://github.com/Febrian-chiperbase/gold-price-prediction-ai.git)
    cd gold-price-prediction-ai/analisis_emas.py
    python analisis_emas.py
    ```

2.  **Buat dan aktifkan lingkungan virtual Python:**
    ```bash
    python3 -m venv env_emas
    source env_emas/bin/activate
    # Untuk Windows: env_emas\Scripts\activate
    ```

3.  **Instal dependensi/pustaka yang diperlukan:**
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Instalasi `prophet` dan `tensorflow` mungkin memerlukan waktu dan beberapa dependensi sistem (seperti `build-essential` di Linux). Jika ada masalah, silakan merujuk ke dokumentasi resmi masing-masing pustaka.*

4.  **Jalankan skrip analisis:**
    Skrip utama berada di direktori `src/`.
    ```bash
    python src/analisis_emas.py
    ```

5.  **Lihat Hasil:**
    * Output analisis dan prediksi akan ditampilkan di terminal.
    * Plot hasil prediksi akan disimpan sebagai file `.png` di direktori `images/`.

## Hasil Contoh

Berikut adalah contoh plot hasil prediksi menggunakan model LSTM:

![Analisis Harga Emas Menggunakan LSTM](images/gold_prediction_lstm.png)

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan lakukan *fork* repositori, buat *branch* baru untuk fitur atau perbaikan Anda, dan kemudian buat *Pull Request*.

## Disclaimer

Proyek ini dibuat untuk tujuan edukasi dan eksplorasi teknologi AI dalam analisis pasar keuangan. Informasi dan prediksi yang dihasilkan oleh model ini tidak boleh dianggap sebagai saran finansial. Perdagangan komoditas memiliki risiko yang signifikan. Lakukan riset Anda sendiri sebelum membuat keputusan investasi.

## Lisensi

Proyek ini dilisensikan di bawah apache2.0 - lihat file `LICENSE` 
