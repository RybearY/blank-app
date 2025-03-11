import librosa
import zipfile
import soundfile as sf
from soundfile import LibsndfileError
from pydub import AudioSegment
from copy import deepcopy
from io import BytesIO

def convert_audio(input_buffer, target_sr, target_bit_depth, target_format='WAV', mono=True):
    output_buffer = BytesIO()
    try:
        try:
            y, sr = librosa.load(deepcopy(input_buffer), sr=None, mono=mono) # sr=None으로 원본 샘플레이트 유지
        except LibsndfileError:
            wav_buffer = BytesIO()
            audio_file = AudioSegment.from_file(deepcopy(input_buffer))
            audio_file.export(wav_buffer, format="wav")
            y, sr = librosa.load(wav_buffer, sr=None, mono=mono)

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

def get_audio_zip_file(audio_buffer_list : list[BytesIO]):
    zip_buffer = BytesIO()
    error_files = []
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for audio_buffer in audio_buffer_list:
                try:
                    filename, buffer_data = audio_buffer
                    zipf.writestr(filename, buffer_data.getbuffer())
                except Exception as e:
                    error_files.append(f"{audio_buffer[0]}: {e}")
                    continue
        return (zip_buffer, error_files)
    except Exception as e:
        return (f"Zip Error: {e}", error_files)