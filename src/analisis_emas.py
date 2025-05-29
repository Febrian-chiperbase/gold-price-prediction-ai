# install_dependencies.sh (jalankan ini di terminal Linux Debian Anda sebelum menjalankan script Python)
# sudo apt update
# sudo apt install python3 python3-pip python3-venv build-essential python3-dev -y
# python3 -m venv gold_env
# source gold_env/bin/activate
# pip install pandas numpy scikit-learn tensorflow prophet yfinance matplotlib

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import matplotlib.pyplot as plt
import logging

# Menonaktifkan logging INFO dari TensorFlow (opsional, untuk output yang lebih bersih)
tf.get_logger().setLevel(logging.ERROR)
# Menonaktifkan logging INFO dari cmdstanpy (digunakan oleh Prophet)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)


# --- 1. Pengumpulan Data ---
def fetch_gold_data(ticker='GC=F', start_date='2010-01-01', end_date=None):
    """Mengunduh data harga emas historis."""
    if end_date is None:
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    print(f"Mengunduh data harga emas ({ticker}) dari {start_date} hingga {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print("Tidak ada data yang diunduh. Periksa ticker atau rentang tanggal.")
            return None
        # Pastikan 'Date' adalah index jika belum
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('Date') # Asumsikan ada kolom 'Date' jika bukan index
            
        data = data[['Close']].copy()
        data.dropna(inplace=True)
        print(f"Data berhasil diunduh. Jumlah baris: {len(data)}")
        return data
    except Exception as e:
        print(f"Error saat mengunduh data: {e}")
        return None

# --- 2. Pra-pemrosesan Data ---
def preprocess_data_lstm(data, sequence_length=60):
    """Pra-pemrosesan data untuk model LSTM."""
    print("Melakukan pra-pemrosesan data untuk LSTM...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Pastikan data['Close'] adalah Series 1D sebelum di-reshape
    close_prices = data['Close'].values
    if close_prices.ndim == 1:
        close_prices = close_prices.reshape(-1, 1)
    
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 
    
    print(f"Bentuk X: {X.shape}, Bentuk y: {y.shape}")
    return X, y, scaler

# --- 3. Model AI: LSTM ---
def build_train_lstm_model(X_train, y_train, X_test, y_test, sequence_length, epochs=50, batch_size=32):
    """Membangun dan melatih model LSTM."""
    print("Membangun model LSTM...")
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Melatih model LSTM...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=1, shuffle=False)
    print("Pelatihan LSTM selesai.")
    return model, history

def predict_with_lstm(model, X_train_data_lstm, X_test_data_lstm, data_input_scaled, scaler, sequence_length, future_steps=30):
    """Membuat prediksi menggunakan model LSTM, termasuk prediksi masa depan."""
    print("Membuat prediksi dengan LSTM...")
    
    train_predict_scaled = model.predict(X_train_data_lstm)
    test_predict_scaled = model.predict(X_test_data_lstm)

    train_predict = scaler.inverse_transform(train_predict_scaled)
    test_predict = scaler.inverse_transform(test_predict_scaled)
    
    last_sequence = data_input_scaled[-sequence_length:]
    future_predictions_scaled_list = []
    current_sequence = last_sequence.reshape((1, sequence_length, 1))

    for _ in range(future_steps):
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        future_predictions_scaled_list.append(next_pred_scaled[0,0])
        new_sequence_member = next_pred_scaled.reshape(1,1,1)
        current_sequence = np.append(current_sequence[:,1:,:], new_sequence_member, axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled_list).reshape(-1,1))
    print("Prediksi masa depan LSTM selesai.")
    return train_predict, test_predict, future_predictions

# --- 4. Model Terinspirasi Bayesian: Prophet ---
def train_predict_prophet(data_df_input, future_periods=30, test_size=0.2):
    """Melatih model Prophet dan membuat prediksi."""
    print("Mempersiapkan data untuk Prophet...")
    if not isinstance(data_df_input, pd.DataFrame) or data_df_input.empty:
        print("Error: data_df_input untuk Prophet bukan DataFrame yang valid atau kosong.")
        return None, None, None, None, None

    df_prophet = data_df_input.copy()

    # Memastikan 'Date' (atau index) menjadi kolom 'ds' dan 'Close' menjadi 'y'
    if isinstance(df_prophet.index, pd.DatetimeIndex) and 'ds' not in df_prophet.columns :
        df_prophet.reset_index(inplace=True) # Index menjadi kolom, biasanya bernama 'Date' atau 'index'
    
    if 'Date' in df_prophet.columns and 'ds' not in df_prophet.columns:
        df_prophet.rename(columns={'Date': 'ds'}, inplace=True)
    elif 'index' in df_prophet.columns and 'ds' not in df_prophet.columns: # Jika hasil reset_index adalah 'index'
         df_prophet.rename(columns={'index': 'ds'}, inplace=True)

    if 'Close' in df_prophet.columns and 'y' not in df_prophet.columns:
        df_prophet.rename(columns={'Close': 'y'}, inplace=True)

    if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
        print(f"Error: Kolom 'ds' atau 'y' tidak dapat disiapkan untuk Prophet. Kolom yang ada: {df_prophet.columns.tolist()}")
        return None, None, None, None, None
        
    try:
        df_prophet['y'] = pd.to_numeric(df_prophet['y'])
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    except Exception as e:
        print(f"Error saat konversi tipe data ds/y untuk Prophet: {e}")
        return None, None, None, None, None

    df_prophet.dropna(subset=['ds', 'y'], inplace=True)
    if df_prophet.empty:
        print("Error: DataFrame untuk Prophet kosong setelah dropna ds/y.")
        return None, None, None, None, None
    
    df_prophet_train_ready = df_prophet[['ds', 'y']] # Hanya pilih kolom yang diperlukan Prophet

    train_size_prophet = int(len(df_prophet_train_ready) * (1 - test_size))
    df_train_prophet = df_prophet_train_ready.iloc[:train_size_prophet]
    df_test_prophet_actual_data_series_y = df_prophet_train_ready['y'].iloc[train_size_prophet:]


    if df_train_prophet.empty or len(df_train_prophet) < 2:
        print(f"Error: Data training untuk Prophet tidak cukup (data: {len(df_train_prophet)}) atau kosong.")
        return None, None, None, None, None
    
    print(f"Membangun dan melatih model Prophet dengan {len(df_train_prophet)} baris data training...")
    model_prophet = Prophet(interval_width=0.95, daily_seasonality=True) 
    try:
        model_prophet.fit(df_train_prophet)
    except Exception as e:
        print(f"Error saat Prophet fit: {e}")
        print(f"  Tipe df_train_prophet['y']: {type(df_train_prophet['y'])}")
        print(f"  dtype df_train_prophet['y']: {df_train_prophet['y'].dtype if hasattr(df_train_prophet['y'], 'dtype') else 'N/A'}")
        print(f"  Contoh data ds (5 baris pertama):\n{df_train_prophet['ds'].head()}")
        print(f"  Contoh data y (5 baris pertama):\n{df_train_prophet['y'].head()}")
        return None, None, None, None, None
    print("Pelatihan Prophet selesai.")

    print("Membuat prediksi dengan Prophet...")
    periods_for_future_dates = len(df_test_prophet_actual_data_series_y) + future_periods
    if periods_for_future_dates <= 0:
        print("Tidak ada periode untuk prediksi masa depan Prophet.")
        return model_prophet, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    future_dates = model_prophet.make_future_dataframe(periods=periods_for_future_dates)
    forecast = model_prophet.predict(future_dates)
    print("Prediksi Prophet selesai.")
    
    forecast_test_period = forecast.iloc[train_size_prophet : len(df_prophet_train_ready)]
    forecast_future_period = forecast.iloc[len(df_prophet_train_ready):]

    return model_prophet, forecast, forecast_test_period, forecast_future_period, df_test_prophet_actual_data_series_y

# --- 5. Evaluasi Model ---
def evaluate_model(y_true, y_pred, model_name="Model"):
    """Mengevaluasi model menggunakan RMSE dan MAE."""
    # Pastikan y_true dan y_pred adalah array NumPy 1D
    y_true_np = np.array(y_true).ravel()
    y_pred_np = np.array(y_pred).ravel()

    if len(y_true_np) != len(y_pred_np):
        print(f"Error evaluasi {model_name}: Panjang y_true ({len(y_true_np)}) dan y_pred ({len(y_pred_np)}) tidak sama.")
        return np.nan, np.nan

    if len(y_true_np) == 0:
        print(f"Error evaluasi {model_name}: Tidak ada data untuk dievaluasi.")
        return np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)
    print(f"Evaluasi {model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    return rmse, mae

# --- 6. Visualisasi ---
def plot_predictions(original_data, train_predict_plot_data, test_predict_plot_data, future_predict_data, model_name="LSTM"):
    """Memvisualisasikan hasil prediksi."""
    plt.figure(figsize=(15, 7))
    
    if not isinstance(original_data.index, pd.DatetimeIndex):
        try:
            original_data.index = pd.to_datetime(original_data.index)
        except Exception as e:
            print(f"Error converting original_data.index to DatetimeIndex: {e}")
            return

    actual_prices = original_data['Close'].values.squeeze()
    
    plt.plot(original_data.index, actual_prices, label='Harga Aktual Keseluruhan', color='blue', alpha=0.7)

    if train_predict_plot_data is not None and not np.all(np.isnan(train_predict_plot_data)):
        plt.plot(original_data.index, train_predict_plot_data, 
                 label=f'Prediksi {model_name} (Train)', color='orange', linestyle='--')

    if test_predict_plot_data is not None and not np.all(np.isnan(test_predict_plot_data)):
        plt.plot(original_data.index, test_predict_plot_data, 
                 label=f'Prediksi {model_name} (Test)', color='red', linestyle='--')
    
    if future_predict_data is not None and len(future_predict_data) > 0:
        last_date = original_data.index[-1]
        future_dates_idx = pd.to_datetime([last_date + pd.Timedelta(days=i) for i in range(1, len(future_predict_data) + 1)])
        plt.plot(future_dates_idx, future_predict_data, label=f'Prediksi {model_name} (Masa Depan)', color='purple', linestyle=':')
    
    plt.title(f'Analisis Harga Emas Menggunakan {model_name}')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Emas (Close)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'gold_prediction_{model_name.lower()}.png')
    print(f"Plot disimpan sebagai gold_prediction_{model_name.lower()}.png")
    plt.close()

def plot_prophet_forecast(model, forecast, original_data_prophet_ready_format):
    """Memvisualisasikan hasil prediksi Prophet."""
    fig1 = model.plot(forecast)
    plt.title('Prediksi Harga Emas Menggunakan Prophet')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Emas (y)')
    # Pastikan original_data_prophet_ready_format memiliki kolom 'ds' dan 'y'
    if 'ds' in original_data_prophet_ready_format.columns and 'y' in original_data_prophet_ready_format.columns:
        plt.plot(original_data_prophet_ready_format['ds'], original_data_prophet_ready_format['y'], 'k.', label='Harga Aktual')
    else:
        print("Peringatan: Kolom 'ds' atau 'y' tidak ditemukan di data original untuk plot Prophet.")
    plt.legend()
    fig1.savefig('gold_prediction_prophet_detail.png')
    print("Plot Prophet detail disimpan sebagai gold_prediction_prophet_detail.png")
    plt.close(fig1)

    fig2 = model.plot_components(forecast)
    fig2.savefig('gold_prediction_prophet_components.png')
    print("Plot komponen Prophet disimpan sebagai gold_prediction_prophet_components.png")
    plt.close(fig2)

# --- Main Execution ---
if __name__ == "__main__":
    GOLD_TICKER = 'GC=F'
    START_DATE = '2010-01-01'
    END_DATE = None 
    SEQUENCE_LENGTH_LSTM = 60
    TRAIN_TEST_SPLIT_RATIO = 0.85
    FUTURE_PREDICTION_DAYS = 90

    gold_df = fetch_gold_data(ticker=GOLD_TICKER, start_date=START_DATE, end_date=END_DATE)

    if gold_df is not None and not gold_df.empty:
        # --- Analisis dengan LSTM ---
        print("\n--- Memulai Analisis dengan LSTM ---")
        X_lstm, y_lstm, scaler_lstm = preprocess_data_lstm(gold_df, SEQUENCE_LENGTH_LSTM)
        
        train_size_lstm = int(len(X_lstm) * TRAIN_TEST_SPLIT_RATIO)
        X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]
        print(f"Data LSTM: X_train: {X_train_lstm.shape}, y_train: {y_train_lstm.shape}, X_test: {X_test_lstm.shape}, y_test: {y_test_lstm.shape}")

        if len(X_train_lstm) > 0 and len(X_test_lstm) > 0:
            lstm_model, history_lstm = build_train_lstm_model(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, 
                                                              sequence_length=SEQUENCE_LENGTH_LSTM, epochs=50)
            
            train_predict_lstm, test_predict_lstm, future_predictions_lstm = predict_with_lstm(
                lstm_model,
                X_train_lstm, 
                X_test_lstm,  
                scaler_lstm.transform(gold_df['Close'].values.reshape(-1,1)), 
                scaler_lstm, 
                SEQUENCE_LENGTH_LSTM,
                future_steps=FUTURE_PREDICTION_DAYS
            )
            
            y_test_actual_lstm = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1,1))
            evaluate_model(y_test_actual_lstm, test_predict_lstm, model_name="LSTM")
            
            original_prices_lstm_values = gold_df['Close'].values.squeeze()
            
            train_predict_plot_lstm = np.full_like(original_prices_lstm_values, np.nan, dtype=float) 
            idx_start_train = SEQUENCE_LENGTH_LSTM
            idx_end_train = idx_start_train + len(train_predict_lstm)
            if idx_end_train <= len(train_predict_plot_lstm) and idx_start_train < len(train_predict_plot_lstm):
                 train_predict_plot_lstm[idx_start_train:idx_end_train] = train_predict_lstm.reshape(-1) 
            else:
                print(f"Peringatan: Indeks ({idx_start_train}:{idx_end_train}) di luar batas untuk train_predict_plot_lstm (len: {len(train_predict_plot_lstm)}).")

            test_predict_plot_lstm = np.full_like(original_prices_lstm_values, np.nan, dtype=float)
            idx_start_test = idx_end_train 
            idx_end_test = idx_start_test + len(test_predict_lstm)

            if idx_start_test < idx_end_test and idx_end_test <= len(test_predict_plot_lstm) and idx_start_test < len(test_predict_plot_lstm):
                 test_predict_plot_lstm[idx_start_test:idx_end_test] = test_predict_lstm.reshape(-1)
            else:
                available_space = len(test_predict_plot_lstm) - idx_start_test
                if available_space > 0 and idx_start_test < len(test_predict_plot_lstm) :
                    test_predict_plot_lstm[idx_start_test:] = test_predict_lstm[:available_space].reshape(-1)
                    print(f"Peringatan: Prediksi testing LSTM dipotong agar muat ({available_space} poin).")
                elif idx_start_test >= len(test_predict_plot_lstm):
                     print(f"Peringatan: Indeks awal prediksi testing LSTM ({idx_start_test}) di luar batas plot (len: {len(test_predict_plot_lstm)}). Tidak diplot.")
                else:
                    print("Peringatan: Error pada penentuan indeks plot prediksi testing LSTM.")
            
            plot_predictions(gold_df, train_predict_plot_lstm, test_predict_plot_lstm, future_predictions_lstm.reshape(-1), model_name="LSTM")

            print("\nPrediksi Harga Emas Beberapa Hari ke Depan (LSTM):")
            last_actual_date = gold_df.index[-1]
            for i, pred in enumerate(future_predictions_lstm.flatten()):
                print(f"  {(last_actual_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')}: {pred:.2f}")
        else:
            print("Data tidak cukup untuk melatih atau menguji model LSTM setelah pemisahan.")

        # --- Analisis dengan Prophet ---
        print("\n--- Memulai Analisis dengan Prophet ---")
        # Input untuk Prophet adalah DataFrame dengan kolom 'Date' (atau index) dan 'Close'
        # Fungsi train_predict_prophet akan menangani renaming menjadi 'ds' dan 'y'
        prophet_input_df_main = gold_df.copy() 
        
        prophet_test_ratio = len(y_test_lstm) / len(y_lstm) if len(y_lstm) > 0 else 0.2 
        
        if not prophet_input_df_main.empty:
            prophet_results = train_predict_prophet(
                prophet_input_df_main,
                future_periods=FUTURE_PREDICTION_DAYS,
                test_size = prophet_test_ratio
            )
            
            if prophet_results and prophet_results[0] is not None:
                prophet_model, prophet_forecast, prophet_forecast_test_period, prophet_forecast_future_period, prophet_y_test_actual = prophet_results
                
                if prophet_y_test_actual is not None and not prophet_y_test_actual.empty and \
                   prophet_forecast_test_period is not None and not prophet_forecast_test_period.empty:
                    evaluate_model(prophet_y_test_actual.values, prophet_forecast_test_period['yhat'].values, model_name="Prophet (Test Period)")
                else:
                    print("Data aktual atau prediksi test Prophet kosong/tidak valid, evaluasi dilewati.")

                # Siapkan data original untuk plot Prophet
                # df_prophet_train_ready dari dalam fungsi train_predict_prophet sudah punya format ds, y
                # Kita perlu DataFrame original yang sesuai dengan format yang di-fit oleh Prophet
                original_data_prophet_plot = prophet_input_df_main.reset_index() # Pastikan 'Date' jadi kolom
                if 'Date' in original_data_prophet_plot.columns and 'ds' not in original_data_prophet_plot.columns:
                     original_data_prophet_plot.rename(columns={'Date': 'ds'}, inplace=True)
                elif 'index' in original_data_prophet_plot.columns and 'ds' not in original_data_prophet_plot.columns: # Jika hasil reset_index adalah 'index'
                    original_data_prophet_plot.rename(columns={'index': 'ds'}, inplace=True)

                if 'Close' in original_data_prophet_plot.columns and 'y' not in original_data_prophet_plot.columns:
                    original_data_prophet_plot.rename(columns={'Close': 'y'}, inplace=True)


                if prophet_forecast is not None and not prophet_forecast.empty and \
                   'ds' in original_data_prophet_plot and 'y' in original_data_prophet_plot:
                     plot_prophet_forecast(prophet_model, prophet_forecast, original_data_prophet_plot[['ds','y']])
                else:
                    print("Forecast Prophet atau data original untuk plot Prophet kosong/format salah, plotting dilewati.")
                
                if prophet_forecast_future_period is not None and not prophet_forecast_future_period.empty:
                    print("\nPrediksi Harga Emas Beberapa Hari ke Depan (Prophet):")
                    future_prophet_predictions = prophet_forecast_future_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    for index, row in future_prophet_predictions.iterrows():
                        print(f"  {row['ds'].strftime('%Y-%m-%d')}: Prediksi={row['yhat']:.2f} (Low: {row['yhat_lower']:.2f}, High: {row['yhat_upper']:.2f})")
                else:
                    print("Tidak ada prediksi masa depan dari Prophet.")
            else:
                print("Gagal melatih atau membuat prediksi dengan Prophet.")
        else:
            print("Data input untuk Prophet tidak valid atau kosong, analisis Prophet dilewati.")

    else:
        print("Tidak dapat melanjutkan analisis karena gagal mendapatkan data.")

    print("\nAnalisis selesai.")
    print("Untuk melihat plot, buka file .png yang telah disimpan di direktori yang sama dengan script ini.")
