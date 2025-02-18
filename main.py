import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from io import BytesIO

st.set_page_config(page_title="Blank App", layout="wide")

st.title("Audio Requirements Validator")
st.write("업로드된 오디오 파일의 속성을 검증하고, 요구사항과 비교하여 결과를 제공합니다.")

if not st.session_state.get("disabled"):
    st.session_state.disabled = False
if not st.session_state.get("start_button_clicked"):
    st.session_state["start_button_clicked"] = False

# 파일 요구사항 설정 (기존 코드와 동일)
st.sidebar.header("파일 요구사항 설정")
required_format = st.sidebar.selectbox("Format", ["WAV", "MP3", "AAC"], disabled=st.session_state.disabled)
required_channels = st.sidebar.selectbox("Channels", [1, 2], disabled=st.session_state.disabled)
required_sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [44100, 48000, 96000, 192000], disabled=st.session_state.disabled)
required_bit_depth = st.sidebar.selectbox("Bit Depth", [16, 24, 32, "32 (float)", "64 (float)"], disabled=st.session_state.disabled)
required_noise_floor = st.sidebar.slider("Noise Floor (dBFS)", min_value=-100, max_value=0, value=-60, disabled=st.session_state.disabled)
required_stereo_status = st.sidebar.selectbox(
    "Stereo Status", ["Dual Mono", "Mono", "True Stereo", "Joint Stereo"], disabled=st.session_state.disabled
)
sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
with sidebar_col1:
    required_start_button = st.sidebar.button("Start", key="start_button", use_container_width=True)
with sidebar_col2:
    required_new_button = st.sidebar.button("New", key="new_button", use_container_width=True)

if required_start_button:
    st.session_state["start_button_clicked"] = True
    st.session_state.disabled = True
    st.rerun()
if required_new_button:
    st.session_state["start_button_clicked"] = False
    st.session_state["disabled"] = False
    st.rerun()
    
if st.session_state["start_button_clicked"] == True:
    st.header("1. 파일 업로드", divider="red")
    # 여러 파일 업로드 위젯 (기존 코드와 동일)
    uploaded_files = st.file_uploader(
        "오디오 파일을 업로드하세요 (WAV 형식 등 여러 개 가능)", type=["wav", "mp3", "aac"], accept_multiple_files=True
    )

    # 오디오 처리 Generator 함수
    def process_audio_files_generator(uploaded_files):
        for uploaded_file in uploaded_files:
            # Buffer 객체 얻기
            audio_buffer = BytesIO(uploaded_file.getvalue())

            # 오디오 속성 분석 함수 (buffer 객체 입력으로 수정)
            def get_audio_properties_from_buffer(buffer):
                try:
                    data, samplerate = sf.read(buffer) # soundfile은 buffer 객체에서 읽기 지원
                    bit_depth = 'Unknown'
                    if data.dtype == 'int16':
                        bit_depth = 16
                    elif data.dtype == 'int24':
                        bit_depth == 24
                    elif data.dtype == 'int32':
                        bit_depth = 32
                    elif data.dtype == 'float32':
                        bit_depth = '32 (float)'
                    elif data.dtype == 'float64':
                        bit_depth = '64 (float)'

                    channels = data.shape[1] if len(data.shape) > 1 else 1
                    duration = len(data) / samplerate

                    return {
                        "Sample Rate": samplerate,
                        "Channels": channels,
                        "Bit Depth": bit_depth,
                        "Duration (seconds)": round(duration, 2)
                    }
                except Exception as e: # soundfile 오류 처리 (특히 MP3 파일)
                    return { # 기본적인 정보만 반환 (샘플레이트, 채널 정보는 불확실)
                        "Sample Rate": "Error",
                        "Channels": "Error",
                        "Bit Depth": "Error",
                        "Duration (seconds)": "Error (Processing Failed)"
                    }

            def calculate_noise_floor_from_buffer(buffer):
                try:
                    data, samplerate = sf.read(buffer) # soundfile은 buffer 객체에서 읽기 지원
                    if len(data.shape) > 1:  # 스테레오 파일인 경우 첫 번째 채널만 사용
                        data = data[:, 0]

                    # 2. 오디오 데이터 정규화 (필요한 경우) - 이미 -1 ~ +1 범위라고 가정
                    # data = data / np.max(np.abs(data)) # 최대값으로 정규화 (이미 정규화되어 있다고 가정)

                    # 3. RMS (Root Mean Square) 계산
                    rms = np.sqrt(np.mean(data**2))

                    # 4. RMS 레벨을 dBFS (Full Scale 데시벨) 단위로 변환
                    if rms > 0: # 0으로 나누는 오류 방지
                        noise_floor_dbfs = 20 * np.log10(rms)
                    else:
                        noise_floor_dbfs = -np.inf # 0 RMS는 -무한대 dBFS

                    return round(noise_floor_dbfs, 2)

                except Exception as e:
                    print(f"오류 발생: {e}")
                    return None


            def check_stereo_status_from_buffer(buffer):
                try:
                    data, _ = sf.read(buffer) # soundfile은 buffer 객체에서 읽기 지원
                    if len(data.shape) == 1:
                        return "Mono"
                    elif np.array_equal(data[:, 0], data[:, 1]):
                        return "Dual Mono"
                    else:
                        return "True Stereo"
                except Exception as e:
                    return f"Unknown (Error) {e}" # 또는 None 등 오류 표시

            # 기본 속성 가져오기 (buffer 객체 전달)
            properties = get_audio_properties_from_buffer(deepcopy(audio_buffer))
            noise_floor_val = calculate_noise_floor_from_buffer(deepcopy(audio_buffer))
            stereo_status = check_stereo_status_from_buffer(deepcopy(audio_buffer))

            # 요구사항과 비교 (기존 코드와 동일, properties 및 검증 값은 buffer 기반 함수에서 얻음)
            matches_format = uploaded_file.name.lower().endswith(required_format.lower())
            matches_channels = properties["Channels"] == required_channels if properties["Channels"] != "Error" else False
            matches_sample_rate = properties["Sample Rate"] == required_sample_rate if properties["Sample Rate"] != "Error" else False
            matches_bit_depth = str(properties["Bit Depth"]) == str(required_bit_depth) if properties["Bit Depth"] != "Error" else False
            matches_noise_floor = noise_floor_val >= required_noise_floor if isinstance(noise_floor_val, (int, float)) else False
            matches_stereo_status = stereo_status == required_stereo_status if stereo_status != "Unknown (Error)" else False

            matches_all = all(
                [
                    matches_format,
                    matches_channels,
                    matches_sample_rate,
                    matches_bit_depth,
                    matches_noise_floor,
                    matches_stereo_status,
                ]
            )

            # 결과 yield (결과 데이터 생성 방식은 기존 코드와 유사)
            yield {
                "Name": uploaded_file.name,
                "Format": required_format if matches_format else f"Mismatch ({uploaded_file.name.split('.')[-1].upper()})",
                "Sample Rate": f"{properties['Sample Rate']} Hz" if properties["Sample Rate"] != "Error" else "Error",
                "Bit Depth": properties["Bit Depth"] if properties["Bit Depth"] != "Error" else "Error",
                "Channel(1)": properties["Channels"] if properties["Channels"] != "Error" else "Error",
                "Channel(2)": stereo_status,
                "Noise Floor (dBFS)": noise_floor_val if isinstance(noise_floor_val, (int, float)) else "Error",
                "Time (sec)": properties["Duration (seconds)"] if properties["Duration (seconds)"] != "Error (Processing Failed)" else "Error",
                "Valid": "O" if matches_all else "X",
            }

    if uploaded_files:
        results = []  # 결과를 저장할 리스트 (generator 결과를 모으기 위해)

        # Generator 함수 호출 및 결과 수집
        for result in process_audio_files_generator(uploaded_files):
            results.append(result)

        # **결과를 표 형태로 출력 (기존 코드와 동일)**
        st.header("2. 파일 검증 결과", divider="red")
        st.subheader("2.1 결과표", divider="orange")
        df_results = pd.DataFrame(results).reset_index(drop=True)

        def highlight_rows(row):
            color = 'background-color: lightgreen;color: black;' if row["Valid"] == "O" else 'background-color: lightcoral;color: black;'
            return [color] * len(row)

        styled_df_results = df_results.style.apply(highlight_rows, axis=1).format(precision=2)
        st.table(styled_df_results)

        st.subheader("2.2 미리듣기 및 파형", divider="orange")
        for idx, uploaded_file in enumerate(uploaded_files):
            # **미리 듣기 (기존 코드와 동일)**
            st.write(f"##### {idx}. {uploaded_file.name}")
            # **노이즈 음파 시각화 (buffer 객체 사용, torchaudio 사용하는 예시)**
            col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
            with col1:
                st.audio(uploaded_file)
            with col2:
                try:
                    audio_buffer = BytesIO(uploaded_file.getvalue())
                    data, samplerate = sf.read(audio_buffer)

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
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"음파 시각화 오류: (음파 시각화 실패. 파일 형식 또는 코덱을 확인하세요.)")
            st.divider()


    else:
        st.info("파일을 업로드하면 결과가 표시됩니다.")