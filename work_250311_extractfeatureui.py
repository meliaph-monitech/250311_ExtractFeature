import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft, fftfreq
from io import BytesIO

def extract_advanced_features(signal, fs=1):
    """Extracts various statistical and frequency-based features from a signal."""
    features = {}
    n = len(signal)
    if n == 0:
        return None

    # Statistical features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['var'] = np.var(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['median'] = np.median(signal)
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)

    # Frequency domain features
    signal_fft = fft(signal)
    psd = np.abs(signal_fft) ** 2
    freqs = fftfreq(n, 1/fs)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    features['dominant_frequency'] = positive_freqs[np.argmax(positive_psd)] if len(positive_psd) > 0 else 0

    return features

st.title("Feature Extraction from Bead Segmentation")

# File upload section
uploaded_zip = st.file_uploader("Upload ZIP file containing CSVs", type=['zip'])

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        extracted_folder = "extracted_data"
        os.makedirs(extracted_folder, exist_ok=True)
        zip_ref.extractall(extracted_folder)

    csv_files = [os.path.join(extracted_folder, f) for f in os.listdir(extracted_folder) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the uploaded ZIP.")
    else:
        st.success(f"Loaded {len(csv_files)} CSV files.")
        sample_df = pd.read_csv(csv_files[0])
        filter_column = st.selectbox("Select Filter Column", sample_df.columns)
        threshold = st.number_input("Enter Threshold Value", value=0.0)

        if st.button("Segment Beads"):
            segmented_data = []
            metadata = []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(csv_files):
                df = pd.read_csv(file)
                start_indices, end_indices = [], []
                signal = df[filter_column].to_numpy()
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
                
                for bead_num, (start, end) in enumerate(zip(start_indices, end_indices), start=1):
                    metadata.append({
                        "file": file,
                        "bead_number": bead_num,
                        "start_index": start,
                        "end_index": end
                    })
                    bead_segment = df.iloc[start:end + 1]
                    segmented_data.append({
                        "data": bead_segment,
                        "file": file,
                        "bead_number": bead_num,
                        "start_index": start,
                        "end_index": end
                    })
                
                progress_bar.progress((idx + 1) / len(csv_files))
            
            st.success("Bead segmentation complete!")

        # Feature selection
        all_features = list(extract_advanced_features(np.array([1, 2, 3])).keys())
        selected_features = st.multiselect("Select Features to Extract", options=["All"] + all_features, default="All")
        
        if "All" in selected_features:
            selected_features = all_features

        if st.button("Extract Features"):
            features_list = []
            progress_bar = st.progress(0)
            
            for i, segment in enumerate(segmented_data):
                signal = segment['data'].iloc[:, 0].values
                feature_dict = extract_advanced_features(signal)
                if feature_dict:
                    feature_dict = {key: feature_dict[key] for key in selected_features}
                    feature_dict.update({
                        "file": segment['file'],
                        "bead_number": segment['bead_number']
                    })
                    features_list.append(feature_dict)
                progress_bar.progress((i + 1) / len(segmented_data))
            
            features_df = pd.DataFrame(features_list)
            features_df['file_name'] = features_df['file'].str.split('/').str[-1]
            features_df = features_df.rename(columns={'file': 'file_dir'})
            
            st.success("Feature extraction complete!")
            st.dataframe(features_df.head())
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                features_df.to_excel(writer, index=False, sheet_name='Features')
            st.download_button(
                label="Download Results",
                data=output.getvalue(),
                file_name="extracted_features.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
