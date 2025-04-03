import streamlit as st
import zipfile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np
from collections import defaultdict

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the ZIP file.")
        st.stop()
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 20
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    cv = std_val / mean_val if mean_val != 0 else 0
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))
    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))
    x = np.arange(n)
    slope, _ = np.polyfit(x, signal, 1)
    rolling_window = max(10, n // 10)
    rolling_mean = np.convolve(signal, np.ones(rolling_window) / rolling_window, mode='valid')
    moving_average = np.mean(rolling_mean)
    threshold = 3 * std_val
    outlier_count = np.sum(np.abs(signal - mean_val) > threshold)
    extreme_event_duration = 0
    current_duration = 0
    for value in signal:
        if np.abs(value - mean_val) > threshold:
            current_duration += 1
        else:
            extreme_event_duration = max(extreme_event_duration, current_duration)
            current_duration = 0
    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, 
            spectral_entropy, autocorrelation, rms, 
            slope, moving_average, outlier_count, extreme_event_duration]

st.set_page_config(layout="wide")
st.title("Laser Welding Feature Correlation Analysis")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        analysis_level = st.radio("Select correlation analysis level:",
                                  ["Single CSV", "Single Bead Number", "Multiple CSVs - Specific Bead", "All Data"])
        selected_csvs = st.multiselect("Select CSV files", csv_files) if analysis_level != "Single CSV" else [st.selectbox("Select a CSV file", csv_files)]
        selected_beads = st.multiselect("Select Bead Numbers", list(range(1, 31))) if analysis_level in ["Single Bead Number", "Multiple CSVs - Specific Bead"] else None
        if st.button("Run Correlation Analysis"):
            with st.spinner("Computing correlation..."):
                feature_data = []
                for file in selected_csvs:
                    df = pd.read_csv(file)
                    if analysis_level in ["Single Bead Number", "Multiple CSVs - Specific Bead"]:
                        df = df[df[filter_column].isin(selected_beads)]
                    segments = df.groupby(filter_column).apply(lambda x: extract_advanced_features(x.iloc[:, 0].values))
                    for idx, features in segments.items():
                        feature_data.append([file, idx] + features)
                feature_columns = ["file", "bead_number"] + [f"Feature_{i}" for i in range(len(features))]
                feature_df = pd.DataFrame(feature_data, columns=feature_columns)
                correlation_matrix = feature_df.iloc[:, 2:].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                st.success("Correlation analysis complete!")
