## ADVANCED FEATURE EXTRACTION
import streamlit as st
import zipfile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

def segment_beads(df, column, threshold):
    start_indices = []
    end_indices = []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 20  # Default feature values

    # Handle NaN or Inf values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return [0] * 20  # Return default values if data is bad

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

    # FFT calculations
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))

    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))

    # Handle edge cases for np.polyfit
    x = np.arange(n)
    if len(set(signal)) == 1 or len(signal) < 2:  # Constant or too short signal
        slope = 0
    else:
        try:
            slope, _ = np.polyfit(x, signal, 1)
        except np.linalg.LinAlgError:
            slope = 0

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
st.title("Laser Welding Correlation Analysis")

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
        
        if st.button("Segment Beads"):
            with st.spinner("Segmenting beads..."):
                bead_segments = {}
                metadata = []
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    if segments:
                        bead_segments[file] = segments
                        for bead_num, (start, end) in enumerate(segments, start=1):
                            metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata
        
        bead_numbers = st.text_input("Enter bead numbers (comma-separated)")
        if st.button("Select Beads") and "metadata" in st.session_state:
            selected_beads = [int(b.strip()) for b in bead_numbers.split(",") if b.strip().isdigit()]
            chosen_bead_data = []
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_beads:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    chosen_bead_data.append({"data": bead_segment, "file": entry["file"], "bead_number": entry["bead_number"], "start_index": entry["start_index"], "end_index": entry["end_index"]})
            st.session_state["chosen_bead_data"] = chosen_bead_data
            st.success("Beads selected successfully!")

# Feature selection
feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak", "Energy", "Coefficient of Variation (CV)",
                 "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)", "Slope", "Moving Average",
                 "Outlier Count", "Extreme Event Duration"]
options = ["All"] + feature_names

selected_features = st.multiselect(
    "Select features to use for Correlation Analysis",
    options=options,
    default="All"
)

if "All" in selected_features and len(selected_features) > 1:
    selected_features = ["All"]
elif "All" not in selected_features and len(selected_features) == 0:
    st.error("You must select at least one feature.")
    st.stop()
if "All" in selected_features:
    selected_features = feature_names
selected_indices = [feature_names.index(f) for f in selected_features]

# Pairwise Correlation Analysis
if st.button("Run Correlation Analysis") and "chosen_bead_data" in st.session_state:
    with st.spinner("Running Correlation Analysis..."):
        correlation_results = {}
        for bead_number in sorted(set(seg["bead_number"] for seg in st.session_state["chosen_bead_data"])):
            bead_data = [seg for seg in st.session_state["chosen_bead_data"] if seg["bead_number"] == bead_number]
            signals = [seg["data"].iloc[:, 0].values for seg in bead_data]
            feature_matrix = np.array([extract_advanced_features(signal) for signal in signals])
            feature_matrix = feature_matrix[:, selected_indices]  # Select only the chosen features
            
            # Compute correlation matrix
            correlation_matrix = pd.DataFrame(feature_matrix, columns=selected_features).corr()
            correlation_results[bead_number] = correlation_matrix

        st.session_state["correlation_results"] = correlation_results
        st.success("Correlation Analysis complete!")

# Visualization
if "correlation_results" in st.session_state:
    st.write("## Visualization")
    for bead_number, correlation_matrix in st.session_state["correlation_results"].items():
        st.write(f"### Bead Number {bead_number}")
        
        corr = correlation_matrix
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask for the upper triangle
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=False,
            cbar=True,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        
        # Add circles for correlation values
        for i in range(len(corr)):
            for j in range(i + 1):  # Only plot the lower triangle
                value = corr.iloc[i, j]
                circle_size = abs(value) * 20  # Adjust size multiplier for better visibility
                color = plt.cm.coolwarm((value + 1) / 2)  # Map value to colormap
                ax.add_patch(plt.Circle((j + 0.5, i + 0.5), circle_size / 100, color=color, alpha=0.8))
        
        ax.set_xticks(np.arange(len(corr.columns)) + 0.5)
        ax.set_yticks(np.arange(len(corr.columns)) + 0.5)
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)
        ax.set_title(f"Correlation Heatmap with Circles - Bead {bead_number}", fontsize=14, pad=20)
        
        st.pyplot(fig)
