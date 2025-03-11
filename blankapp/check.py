import streamlit as st
import numpy as np
import soundfile as sf
import filetype
from soundfile import LibsndfileError
from pydub import AudioSegment
from copy import deepcopy
from io import BytesIO

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

def validate_filetype(buffer):
    validate_mimetypes = [
        "audio/mpeg",
        "audio/mp4",
        "video/mp4",
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
    try:
        with sf.SoundFile(deepcopy(buffer), 'r') as f:
            data = f.read()
            subtype = f.subtype
            samplerate = f.samplerate
            channels = f.channels

    except LibsndfileError:
        wav_buffer = BytesIO()
        audio_file = AudioSegment.from_file(deepcopy(buffer))
        audio_file.export(wav_buffer, format="wav")
        with sf.SoundFile(wav_buffer, 'r') as f:
            data = f.read()
            subtype = f.subtype
            samplerate = f.samplerate
            channels = f.channels

    try:
        if 'PCM_' in subtype:
            bit_depth = subtype.replace('PCM_', '')
        else:
            bit_depth = 'Unknown'
        duration = len(data) / samplerate

        return {
            "Sample Rate": samplerate,
            "Channels": channels,
            "Bit Depth": bit_depth,
            "Duration (seconds)": round(duration, 2)
        }
    except Exception as e:
        return { # 기본적인 정보만 반환 (샘플레이트, 채널 정보는 불확실)
            "Sample Rate": "Error",
            "Channels": "Error",
            "Bit Depth": "Error",
            "Duration (seconds)": "Error (Processing Failed)"
        }
    
def calculate_noise_floor_from_buffer(buffer, silence_threshold_db=-60, frame_length=2048, hop_length=512):
    try:
        data, sr = sf.read(deepcopy(buffer))  # 오디오 파일 읽기
    except LibsndfileError:
        wav_buffer = BytesIO()
        audio_file = AudioSegment.from_file(deepcopy(buffer))
        audio_file.export(wav_buffer, format="wav")
        data, sr = sf.read(wav_buffer)
    except Exception as e:
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
        data, _ = sf.read(deepcopy(buffer))
    except LibsndfileError:
        wav_buffer = BytesIO()
        audio_file = AudioSegment.from_file(deepcopy(buffer))
        audio_file.export(wav_buffer, format="wav")
        data, _ = sf.read(wav_buffer)
    except Exception as e:
        return f"Unknown (Error) {e}"

    if len(data.shape) == 1:
        return "Mono"
    elif np.array_equal(data[:, 0], data[:, 1]):
        return "Dual Mono"
    else:
        return "True Stereo"
    

def process_audio_files_generator(uploaded_files, uploaded_file_name = None):
    for uploaded_file in uploaded_files:
        if not uploaded_file_name:
            uploaded_file_name_ = uploaded_file.name
        else:
            uploaded_file_name_ = uploaded_file_name
        # Buffer 객체 얻기
        if isinstance(uploaded_file, BytesIO):
            audio_buffer = uploaded_file
        else:
            audio_buffer = BytesIO(uploaded_file.getvalue())

        # 파일 유효성 검사
        is_valid_file, msg, file_type = validate_filetype(audio_buffer)
        if not is_valid_file:
            st.error(f"[{uploaded_file_name_}]: " + msg)
            yield {
                "Valid": "X",
                "Name": uploaded_file_name_,
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
            matches_format = file_type.lower() == st.session_state["required_format"].lower()
            matches_channels = properties["Channels"] == st.session_state["required_channels"] if properties["Channels"] != "Error" else False
            matches_sample_rate = properties["Sample Rate"] == st.session_state["required_sample_rate"] if properties["Sample Rate"] != "Error" else False
            matches_bit_depth = str(properties["Bit Depth"]) == str(st.session_state["required_bit_depth"]) if properties["Bit Depth"] != "Error" else False
            matches_noise_floor = noise_floor_val < st.session_state["required_noise_floor"] if isinstance(noise_floor_val, (int, float)) else False
            matches_stereo_status = stereo_status == st.session_state["required_stereo_status"] if stereo_status != "Unknown (Error)" else False

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
                    "Valid": "O" if matches_all else "X",
                    "Name": uploaded_file_name_,
                    "Time (sec)": properties["Duration (seconds)"] if properties["Duration (seconds)"] != "Error (Processing Failed)" else "Error",
                    "Format": st.session_state["required_format"] if matches_format else f"{uploaded_file_name_.split('.')[-1].upper()}",
                    "Sample Rate": f"{properties['Sample Rate']} Hz" if properties["Sample Rate"] != "Error" else "Error",
                    "Bit Depth": properties["Bit Depth"] if properties["Bit Depth"] != "Error" else "Error",
                    "Channels": properties["Channels"] if properties["Channels"] != "Error" else "Error",
                    "Stereo Status": stereo_status,
                    "Noise Floor (dBFS)": noise_floor_val if isinstance(noise_floor_val, (int, float)) else "Error",
                }