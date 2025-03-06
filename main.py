import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import filetype
from copy import deepcopy
from io import BytesIO

st.set_page_config(page_title="Blank App", layout="wide")

def validate_filetype(buffer):
    validate_mimetypes = [
        "audio/mpeg",
        "audio/x-wav",
        "audio/x-aiff",
        "audio/x-flac",
        "audio/ogg"
    ]
    kind = filetype.guess(buffer)
    if kind is None:
        return False, "The file format is not recognized."
    mime_type = kind.mime
    if mime_type not in validate_mimetypes:
        return False, f"파일 형식이 잘못되었거나, 지원하지 않는 형식입니다. (Invalid or unsupported file format.)", mime_type.split("/")[-1].upper()
    return True, "파일 분석 가능 (Analysis possible)", mime_type.split("/")[-1].split('-')[-1].upper()

# 오디오 속성 분석 함수 (buffer 객체 입력으로 수정)
def get_audio_properties_from_buffer(buffer):
    with sf.SoundFile(buffer, 'r') as f:
        try:
            data = f.read()
            subtype = f.subtype
            if 'PCM_' in subtype:
                bit_depth = subtype.replace('PCM_', '')
            else:
                bit_depth = 'Unknown'
            samplerate = f.samplerate
            channels = f.channels
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

def calculate_noise_floor_from_buffer(buffer, silence_threshold_db=-60, frame_length=2048, hop_length=512):
    try:
        data, sr = sf.read(buffer)  # 오디오 파일 읽기
    except sf.LibsndfileError as e:
        print(f"Error reading audio file: {e}")
        return None

    # 다채널 오디오라면, 모노로 변환 (선택 사항)
    if data.ndim > 1:
        y = np.mean(data, axis=1)
    else:
        y = data

    # 프레임 단위로 RMS 계산
    num_frames = (len(y) - frame_length) // hop_length + 1
    rms_values = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = y[start:end]
        rms_values[i] = np.sqrt(np.mean(frame**2))

    # dB 스케일로 변환 (무음 구간 찾기 위해)
    epsilon = 1e-12
    rms_db = 20 * np.log10((rms_values / np.max(np.abs(y))) + epsilon)

    # 무음 구간 찾기
    silent_frames = rms_db < silence_threshold_db

    # 무음 구간 RMS 평균 (진폭 스케일) -> dBFS로 변환
    if np.any(silent_frames):
        noise_floor_rms = np.mean(rms_values[silent_frames])
        noise_floor_dbfs = 20 * np.log10(noise_floor_rms / np.max(np.abs(y)) + epsilon)
    else:
        print("Warning: No silent frames found. Using minimum RMS.")
        noise_floor_rms = np.min(rms_values)
        noise_floor_dbfs = 20 * np.log10(noise_floor_rms / np.max(np.abs(y)) + epsilon)

    return round(noise_floor_dbfs, 2)

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
    
def convert_audio(input_buffer, target_sr, target_bit_depth, target_format='WAV', mono=True):
    output_buffer = BytesIO()
    try:

        y, sr = librosa.load(input_buffer, sr=None, mono=mono) # sr=None으로 원본 샘플레이트 유지
        print(sr)
        if sr != target_sr:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type='soxr_vhq')
        else:
            y_resampled = y # 샘플레이트가 이미 목표 샘플레이트와 같으면 리샘플링 생략

        if target_format == "WAV" or target_format == "FLAC":
            subtype = f"PCM_{target_bit_depth}"
        else:
            subtype = None

        sf.write(output_buffer, y_resampled, target_sr, format=target_format, subtype=subtype) # WAV, FLAC: PCM_24 권장
        output_buffer.seek(0)

        return output_buffer
    except Exception as e:
        print(f"오류 발생: {e}")

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
    "Stereo Status", ["Dual Mono", "Mono", "True Stereo", "Joint Stereo"], disabled=st.session_state.disabled
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
        "오디오 파일(wav, mp3, flac)을 업로드하세요. (Upload audio files (wav, mp3, flac).)", type=["wav", "mp3", "flac"], accept_multiple_files=True
    )

    # 오디오 처리 Generator 함수
    def process_audio_files_generator(uploaded_files):
        for uploaded_file in uploaded_files:
            # Buffer 객체 얻기
            if isinstance(uploaded_file, BytesIO):
                audio_buffer = uploaded_file
            else:
                audio_buffer = BytesIO(uploaded_file.getvalue())

            # 파일 유효성 검사
            is_valid_file, msg, file_type = validate_filetype(audio_buffer)
            if not is_valid_file:
                st.error(f"[{uploaded_file.name}]: " + msg)
                yield {
                    "Valid": "X",
                    "Name": uploaded_file.name,
                    "Time (sec)": "Error",
                    "Format": file_type,
                    "Sample Rate": "Error",
                    "Bit Depth": "Error",
                    "Channels": "Error",
                    "Stereo Status": "Error",
                    "Noise Floor (dBFS)": "Error",
                }
            else:

                # 기본 속성 가져오기 (buffer 객체 전달)
                properties = get_audio_properties_from_buffer(deepcopy(audio_buffer))
                noise_floor_val = calculate_noise_floor_from_buffer(deepcopy(audio_buffer))
                stereo_status = check_stereo_status_from_buffer(deepcopy(audio_buffer))

                # 요구사항과 비교 (기존 코드와 동일, properties 및 검증 값은 buffer 기반 함수에서 얻음)
                # matches_format = uploaded_file.name.lower().endswith(required_format.lower())
                matches_format = file_type.lower() == required_format.lower()
                matches_channels = properties["Channels"] == required_channels if properties["Channels"] != "Error" else False
                matches_sample_rate = properties["Sample Rate"] == required_sample_rate if properties["Sample Rate"] != "Error" else False
                matches_bit_depth = str(properties["Bit Depth"]) == str(required_bit_depth) if properties["Bit Depth"] != "Error" else False
                matches_noise_floor = noise_floor_val < required_noise_floor if isinstance(noise_floor_val, (int, float)) else False
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
                if isinstance(uploaded_file, BytesIO):
                    yield {
                        "Valid": "O" if matches_all else "X",
                        "Time (sec)": properties["Duration (seconds)"] if properties["Duration (seconds)"] != "Error (Processing Failed)" else "Error",
                        "Format": required_format if matches_format else f"{uploaded_file.name.split('.')[-1].upper()}",
                        "Sample Rate": f"{properties['Sample Rate']} Hz" if properties["Sample Rate"] != "Error" else "Error",
                        "Bit Depth": properties["Bit Depth"] if properties["Bit Depth"] != "Error" else "Error",
                        "Channels": properties["Channels"] if properties["Channels"] != "Error" else "Error",
                        "Stereo Status": stereo_status,
                        "Noise Floor (dBFS)": noise_floor_val if isinstance(noise_floor_val, (int, float)) else "Error",
                    }
                else:
                    yield {
                        "Valid": "O" if matches_all else "X",
                        "Name": uploaded_file.name,
                        "Time (sec)": properties["Duration (seconds)"] if properties["Duration (seconds)"] != "Error (Processing Failed)" else "Error",
                        "Format": required_format if matches_format else f"{uploaded_file.name.split('.')[-1].upper()}",
                        "Sample Rate": f"{properties['Sample Rate']} Hz" if properties["Sample Rate"] != "Error" else "Error",
                        "Bit Depth": properties["Bit Depth"] if properties["Bit Depth"] != "Error" else "Error",
                        "Channels": properties["Channels"] if properties["Channels"] != "Error" else "Error",
                        "Stereo Status": stereo_status,
                        "Noise Floor (dBFS)": noise_floor_val if isinstance(noise_floor_val, (int, float)) else "Error",
                    }

    if uploaded_files:
        results = []  # 결과를 저장할 리스트 (generator 결과를 모으기 위해)

        # Generator 함수 호출 및 결과 수집
        for result in process_audio_files_generator(uploaded_files):
            results.append(result)

        # **결과를 표 형태로 출력 (기존 코드와 동일)**
        st.header("2. 파일 검증 결과 (File verification results)", divider="red")
        st.subheader("2.1 결과표 (Results table)", divider="orange")
        df_results = pd.DataFrame(results).reset_index(drop=True)

        def highlight_rows(row):
            green = 'background-color: lightgreen;color: black;'
            red = 'background-color: lightcoral;color: black;'

            if row['Valid'] == "O":
                colors = [green] * 6
            elif row["Time (sec)"] == "Error":
                return [red] * len(row)
            else:
                colors = [
                    green if row["Format"] == st.session_state["required_format"] else red,
                    green if row["Sample Rate"] == f'{st.session_state["required_sample_rate"]} Hz' else red,
                    green if row["Bit Depth"] == str(st.session_state["required_bit_depth"]) else red,
                    green if row["Channels"] == st.session_state["required_channels"] else red,
                    green if row["Stereo Status"] == st.session_state["required_stereo_status"] else red,
                    red if row["Noise Floor (dBFS)"] == "Error" or row["Noise Floor (dBFS)"] >= st.session_state["required_noise_floor"] else green
                ]
            return [None] * (len(row) - 6) + colors

        styled_df_results = df_results.style.apply(highlight_rows, axis=1).format(precision=2)
        st.table(styled_df_results)

        st.subheader("2.2 미리듣기 및 파형 (Preview and waveform)", divider="orange")
        for idx, uploaded_file in enumerate(uploaded_files):
            # **미리 듣기 (기존 코드와 동일)**
            st.write(f"##### {idx}. {uploaded_file.name}")
            # **노이즈 음파 시각화 (buffer 객체 사용, torchaudio 사용하는 예시)**
            col1, col2 = st.columns([0.3, 0.7], vertical_alignment="center")
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
                    for changed_result in process_audio_files_generator([deepcopy(output_buffer)]):
                        changed_results.append(changed_result)
                    changed_df = pd.DataFrame(changed_results).reset_index(drop=True).loc[:, each_df_cols]
                    each_df = pd.concat([each_df, changed_df]).reset_index(drop=True)
                
                st.table(each_df.style.apply(highlight_rows, axis=1))
                st.markdown("- 원본 오디오 (Original Audio)")
                st.audio(uploaded_file)
                if len(each_df) > 1:
                    st.markdown("- 변환된 오디오 (Converted Audio)")
                    st.audio(output_buffer)
                
                # st.table(df_results.loc[idx:idx, ["Valid", "Time (sec)", "Format", "Sample Rate", "Bit Depth", "Channels", "Stereo Status", "Noise Floor (dBFS)"]].style.apply(highlight_rows, axis=1))
                # st.write("원본 오디오")
                # st.audio(uploaded_file)
                # if df_results.loc[idx, "Valid"] == "X":
                #     st.write("변환된 오디오")
                #     output_buffer = convert_audio(
                #         input_buffer=deepcopy(audio_buffer),
                #         target_sr=st.session_state["required_sample_rate"],
                #         target_format=st.session_state["required_format"]
                #     )
                #     st.audio(output_buffer)
            with col2:
                try:
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
                    ax.legend(loc="upper right")
                    st.pyplot(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"음파 시각화 오류: 음파 시각화 실패. 파일 형식 또는 코덱을 확인하세요. (Audio visualization error: Failed to visualize waveform. Please check the file format or codec.)")
            st.divider()


    else:
        st.info("파일을 업로드하면 결과가 표시됩니다. (The results will be displayed after you upload a file.)")