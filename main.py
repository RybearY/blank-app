import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import filetype
from soundfile import LibsndfileError
from pydub import AudioSegment
from copy import deepcopy
from io import BytesIO

from blankapp.check import highlight_rows, process_audio_files_generator
from blankapp.convert import convert_audio, get_audio_zip_file

st.set_page_config(page_title="Blank App", layout="wide")

st.title("Audio Requirements Validator")
st.markdown("업로드된 오디오 파일의 속성을 검증하고, 요구사항과 비교하여 결과를 제공합니다. (Verify the attributes of the uploaded audio file, compare them against the requirements, and provide the results.)")

info_col1, info_col2 = st.columns(2)
with info_col1:
    with st.container(border=True):
        st.markdown("##### 사용 방법 (How to use)")
        st.markdown('''1. 파일 요구사항 설정값 선택 (Select the file requirement settings)\n2. Save 버튼 클릭 (Click the Save button)\n3. 오디오 파일 업로드 (Upload audio files)\n4. 결과 확인 (Check the results)''')
        st.markdown("새로운 파일 요구사항 설정 : Reset 버튼 클릭 (New file requirement settings: Click the Reset button)")
with info_col2:
    with st.container(border=True):
        st.markdown("##### 참고 (Note)")
        st.markdown('''업로드하신 파일의 형식이 결과표에 표시된 형식과 일치하지 않는 경우가 있습니다. 파일이 올바르게 변환되었는지, 손상되지 않았는지 확인하신 후 다시 업로드해주세요. 문제가 계속될 경우, 다른 형식으로 변환하여 시도해 보시거나, 원본 파일을 다시 다운로드하여 업로드해 주시기 바랍니다.''')
        st.markdown('''There may be a discrepancy between the format of the file you uploaded and the format displayed in the results table. Please ensure that the file has been converted correctly and is not corrupted before re-uploading. If the issue persists, try converting the file to a different format or re-download the original file and upload it again.''')

if not st.session_state.get("disabled"):
    st.session_state.disabled = False
if not st.session_state.get("start_button_clicked"):
    st.session_state["start_button_clicked"] = False

if not st.session_state.get("required_format"):
    st.session_state["required_format"] = None
if not st.session_state.get("required_channels"):
    st.session_state["required_channels"] = None
if not st.session_state.get("required_sample_rate"):
    st.session_state["required_sample_rate"] = None
if not st.session_state.get("required_bit_depth"):
    st.session_state["required_bit_depth"] = None
if not st.session_state.get("required_noise_floor"):
    st.session_state["required_noise_floor"] = None
if not st.session_state.get("required_stereo_status"):
    st.session_state["required_stereo_status"] = None

# 파일 요구사항 설정 (기존 코드와 동일)
st.sidebar.header("파일 요구사항 설정 (File requirements settings)")
required_format = st.sidebar.selectbox("Format", ["WAV", "MP3", "AAC"], disabled=st.session_state.disabled)
required_channels = st.sidebar.selectbox("Channels", [1, 2], disabled=st.session_state.disabled)
required_sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [44100, 48000, 96000, 192000], disabled=st.session_state.disabled)
required_bit_depth = st.sidebar.selectbox("Bit Depth", [16, 24, 32], disabled=st.session_state.disabled)
required_noise_floor = st.sidebar.slider("Noise Floor (dBFS)", min_value=-100, max_value=0, value=-60, disabled=st.session_state.disabled)
required_stereo_status = st.sidebar.selectbox(
    "Stereo Status", ["Mono", "Dual Mono", "True Stereo", "Joint Stereo"], disabled=st.session_state.disabled
)

sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
with sidebar_col1:
    required_start_button = st.sidebar.button("Save", key="start_button", use_container_width=True)
with sidebar_col2:
    required_new_button = st.sidebar.button("Reset", key="new_button", use_container_width=True)

if required_start_button:
    st.session_state["required_format"] = required_format
    st.session_state["required_channels"] = required_channels
    st.session_state["required_sample_rate"] = required_sample_rate
    st.session_state["required_bit_depth"] = required_bit_depth
    st.session_state["required_noise_floor"] = required_noise_floor
    st.session_state["required_stereo_status"] = required_stereo_status

    st.session_state["audio_zip_files"] = []
    st.session_state["audio_zip_download"] = False

    st.session_state["start_button_clicked"] = True
    st.session_state.disabled = True
    st.rerun()
if required_new_button:
    st.session_state["start_button_clicked"] = False
    st.session_state["disabled"] = False
    st.rerun()
    
if st.session_state["start_button_clicked"] == True:
    st.header("1. 파일 업로드 (1. Upload file)", divider="red")
    # 여러 파일 업로드 위젯 (기존 코드와 동일)
    uploaded_files = st.file_uploader(
        "오디오 파일(wav, mp3, flac)을 업로드하세요. (Upload audio files (wav, mp3, flac, mp4).)", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []  # 결과를 저장할 리스트 (generator 결과를 모으기 위해)

        # Generator 함수 호출 및 결과 수집
        for result in process_audio_files_generator(uploaded_files):
            results.append(result)

        # **결과를 표 형태로 출력 (기존 코드와 동일)**
        st.header("2. 파일 검증 결과 (File verification results)", divider="red")
        st.subheader("2.1 결과표 (Results table)", divider="orange")
        df_results = pd.DataFrame(results).reset_index(drop=True)

        styled_df_results = df_results.style.apply(highlight_rows, axis=1).format(precision=2)
        st.table(styled_df_results)

        st.subheader("2.2 미리듣기 및 파형 (Preview and waveform)", divider="orange")
        for idx, uploaded_file in enumerate(uploaded_files):
            audio_download_files = []
            # **미리 듣기 (기존 코드와 동일)**
            st.write(f"##### {idx}. {uploaded_file.name}")
            # **노이즈 음파 시각화 (buffer 객체 사용, torchaudio 사용하는 예시)**
            col1, col2 = st.columns([0.5, 0.5], vertical_alignment="center")
            
            audio_buffer = BytesIO(uploaded_file.getvalue())
            with col1:
                each_df_cols = ["Valid", "Time (sec)", "Format", "Sample Rate", "Bit Depth", "Channels", "Stereo Status", "Noise Floor (dBFS)"]
                each_df = df_results.loc[idx:idx, each_df_cols]
                if df_results.loc[idx, "Valid"] == "X":
                    changed_results = []
                    output_buffer = convert_audio(
                        input_buffer=deepcopy(audio_buffer),
                        target_sr=st.session_state["required_sample_rate"],
                        target_bit_depth=st.session_state["required_bit_depth"],
                        target_format=st.session_state["required_format"]
                    )
                    for changed_result in process_audio_files_generator([deepcopy(output_buffer)], uploaded_file_name=uploaded_file.name):
                        changed_results.append(changed_result)
                    changed_df = pd.DataFrame(changed_results).reset_index(drop=True).loc[:, each_df_cols]
                    each_df = pd.concat([each_df, changed_df]).reset_index(drop=True)
                
                st.table(each_df.style.apply(highlight_rows, axis=1).format(precision=2))
                st.markdown("- 원본 오디오 (Original Audio)")
                st.audio(uploaded_file)
                if len(each_df) > 1:
                    st.markdown("- 변환된 오디오 (Converted Audio)")
                    st.audio(output_buffer)
                    audio_download_files.append((uploaded_file.name, output_buffer))
                else:
                    audio_download_files.append((uploaded_file.name, BytesIO(uploaded_file.getvalue())))
                

            with col2:
                try:
                    data, samplerate = sf.read(deepcopy(audio_buffer))
                except LibsndfileError:
                    wav_buffer = BytesIO()
                    audio_file = AudioSegment.from_file(deepcopy(audio_buffer))
                    audio_file.export(wav_buffer, format="wav")
                    data, samplerate = sf.read(wav_buffer)
                try:
                    if data.ndim > 1:
                        data = data[:, 0]

                    fig, ax = plt.subplots(figsize=(10, 2))
                    time_axis = np.linspace(0, len(data) / samplerate, num=len(data))
                    ax.plot(time_axis, data, color='royalblue', linewidth=0.7)
                    ax.axhline(y=10**(required_noise_floor/20), color='red', linestyle='--', label="Required Noise Floor")

                    ax.set_xticks(np.arange(0, max(time_axis), step=5))
                    ax.set_title("Waveform with Noise Floor")
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Amplitude")
                    ax.legend(loc="upper right")
                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"음파 시각화 오류: 음파 시각화 실패. 파일 형식 또는 코덱을 확인하세요. (Audio visualization error: Failed to visualize waveform. Please check the file format or codec.)")
            
             

            st.divider()
    else:
        st.info("파일을 업로드하면 결과가 표시됩니다. (The results will be displayed after you upload a file.)")