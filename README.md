# Anomaly Detection in PCG Data using LSTM Autoencoder

This project implements an **LSTM-based Autoencoder** model to detect anomalies in PCG (Phonocardiogram) signals. It processes and analyzes time-series sensor data collected over time and uses machine learning to classify whether the signal represents a **normal** or **anomalous** pattern.

## 📌 Features

- Preprocessing and resampling of PCG data
- Data normalization using `StandardScaler`
- Sequence modeling using **LSTM Autoencoder**
- Anomaly detection based on MAE (Mean Absolute Error) threshold
- Visualization of results using **Seaborn** and **Matplotlib**

## 🧠 Methodology

1. **Data Loading & Preprocessing**
   - Parse and clean timestamp data
   - Resample signal every 10 seconds
   - Handle missing values using mean imputation

2. **Feature Scaling**
   - Normalize PCG signal using `StandardScaler`

3. **Model Training**
   - LSTM Autoencoder trained on normal data
   - Detect anomalies based on MAE (errors between input and reconstruction)

4. **Evaluation**
   - If MAE exceeds a threshold (2.6), data is labeled as **Anomaly**, else **Normal**

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Keras (TensorFlow backend)
- LSTM (Sequential Model)
- Scikit-learn
- Matplotlib, Seaborn

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pcg-anomaly-lstm.git
   cd pcg-anomaly-lstm
Place your PCG CSV files (Test_2.csv, GW_36_1.csv) in the project directory.

Install required libraries:
pip install -r requirements.txt
Run the main script:
python anomaly_detection.py
Output
Histogram of train and test MAE

Classification of test data as Anomaly or Normal

Line plot of normalized PCG signals

FILE STRUCTURE

├── anomaly_detection.py     # Main Python script
├── Test_2.csv               # Input data file
├── GW_36_1.csv              # Input data file
├── TEST_1.csv               # Resampled & Cleaned PCG
├── Modified.csv             # Final formatted data
├── README.md                # Project documentation
