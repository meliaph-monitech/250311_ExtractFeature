import streamlit as st
import pandas as pd
import zipfile
import os
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from scipy.fftpack import fft, fftfreq

# Function to reset session state

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Streamlit UI
st.title("Bead Segmentation & Feature Extraction")

# File uploader
uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"], on_change=reset_session)

if uploaded_file:
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        extract_path = "temp_extracted"
        zip_ref.extractall(extract_path)
        csv_files = [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith(".csv")]
        st.session_state["csv_files"] = csv_files  # Reset stored CSV files
        st.success(f"Loaded {len(csv_files)} CSV files.")

# Select filter column and threshold
if "csv_files" in st.session_state and st.session_state["csv_files"]:
    sample_df = pd.read_csv(st.session_state["csv_files"][0])
    filter_column = st.selectbox("Select filter column", sample_df.columns)
    filter_threshold = st.number_input("Enter filter threshold", value=0.0)
    
    if st.button("Segment Beads"):
        st.session_state["segmented_data"] = []  # Reset segmented data
        st.session_state["metadata"] = []
        for file in st.session_state["csv_files"]:
            df = pd.read_csv(file)
            signal = df[filter_column].to_numpy()
            start_indices, end_indices = [], []
            i = 0
            while i < len(signal):
                if signal[i] > filter_threshold:
                    start = i
                    while i < len(signal) and signal[i] > filter_threshold:
                        i += 1
                    end = i - 1
                    start_indices.append(start)
                    end_indices.append(end)
                else:
                    i += 1
            segments = list(zip(start_indices, end_indices))
            for bead_num, (start, end) in enumerate(segments, start=1):
                st.session_state["metadata"].append({
                    "file": file,
                    "bead_number": bead_num,
                    "start_index": start,
                    "end_index": end
                })
        st.success("Bead segmentation completed!")

# Feature selection
if "metadata" in st.session_state and st.session_state["metadata"]:
    feature_options = [
        "mean", "std", "var", "min", "max", "median", "skewness", "kurtosis",
        "peak_to_peak", "energy", "rms", "power", "crest_factor", "form_factor",
        "pulse_indicator", "margin", "dominant_frequency", "spectral_entropy",
        "mean_band_power", "max_band_power", "sum_total_band_power", "peak_band_power",
        "var_band_power", "std_band_power", "skewness_band_power", "kurtosis_band_power",
        "relative_spectral_peak_per_band", "peak_count", "zero_crossing_rate",
        "slope", "outlier_count", "extreme_event_duration"
    ]
    selected_features = st.multiselect("Select features to extract", ["All"] + feature_options)
    if "All" in selected_features:
        selected_features = feature_options
    
    if st.button("Extract Features"):
        extracted_features = []
        progress_bar = st.progress(0)
        for i, entry in enumerate(st.session_state["metadata"]):
            df = pd.read_csv(entry["file"])
            signal = df.iloc[entry["start_index"]:entry["end_index"] + 1, 0].values
            features = {}
            if "mean" in selected_features:
                features["mean"] = np.mean(signal)
            if "std" in selected_features:
                features["std"] = np.std(signal)
            if "var" in selected_features:
                features["var"] = np.var(signal)
            if "min" in selected_features:
                features["min"] = np.min(signal)
            if "max" in selected_features:
                features["max"] = np.max(signal)
            if "median" in selected_features:
                features["median"] = np.median(signal)
            if "skewness" in selected_features:
                features["skewness"] = skew(signal)
            if "kurtosis" in selected_features:
                features["kurtosis"] = kurtosis(signal)
            features.update({"bead_number": entry["bead_number"], "file": entry["file"]})
            extracted_features.append(features)
            progress_bar.progress((i + 1) / len(st.session_state["metadata"]))
        
        features_df = pd.DataFrame(extracted_features)
        features_df["file_name"] = features_df["file"].str.split("/").str[-1]
        features_df = features_df.rename(columns={"file": "file_dir"})
        st.session_state["features_df"] = features_df
        st.success("Feature extraction completed!")

# Download button
if "features_df" in st.session_state:
    st.download_button(
        label="Download Results", 
        data=st.session_state["features_df"].to_csv(index=False),
        file_name="extracted_features.csv",
        mime="text/csv"
    )
