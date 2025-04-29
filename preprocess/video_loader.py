import os
import cv2
import subprocess
import wave
import contextlib
from utils.logging_utils import logger

def extract_audio(video_path, output_audio_path):
    logger.info(f"Извлечение аудио из {video_path}...")
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"Аудио сохранено в {output_audio_path}")

def extract_frames(video_path, frames_dir, frame_interval_sec):
    logger.info(f"Извлечение фреймов из {video_path} каждые {frame_interval_sec} секунды...")
    
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Длина видео: {duration:.2f} секунд")
    logger.info(f"Частота кадров (FPS): {fps:.2f}")

    frame_interval = int(fps * frame_interval_sec)
    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(frames_dir, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Извлечено {saved_idx} фреймов в {frames_dir}")

def get_audio_duration(audio_path):
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def preprocess_video(video_path, output_dir, mode="balanced"):
    """
    :param video_path: Путь к видео
    :param output_dir: Папка для сохранения результатов
    :param mode: "fast" (3 сек), "balanced" (1.5 сек), "detailed" (1 сек)
    """

    mode_to_interval = {
        "fast": 3.0,
        "balanced": 1.5,
        "detailed": 1.0,
    }

    if mode not in mode_to_interval:
        raise ValueError(f"Неверный режим '{mode}'. Допустимые: {list(mode_to_interval.keys())}")

    frame_interval_sec = mode_to_interval[mode]

    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, "audio.wav")
    frames_dir = os.path.join(output_dir, "frames")

    extract_audio(video_path, audio_path)
    extract_frames(video_path, frames_dir, frame_interval_sec)

    audio_duration = get_audio_duration(audio_path)
    logger.info(f"Длительность извлечённого аудио: {audio_duration:.2f} секунд")

    return {
        "audio_path": audio_path,
        "frames_dir": frames_dir,
        "audio_duration": audio_duration
    }
