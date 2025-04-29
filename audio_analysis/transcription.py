import os
import json
import whisperx
import torch
import gc
from utils.logging_utils import logger
import matplotlib.pyplot as plt


os.environ["PATH"] = r"C:\Program Files\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin;" + os.environ["PATH"]

def transcribe_audio(audio_path, device="cuda", compute_type="float16", batch_size=4, hf_token=None):
    if hf_token is None:
        raise ValueError("HuggingFace токен обязателен для выполнения диаризации!")

    logger.info(f"Загрузка модели WhisperX...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    logger.info(f"Загрузка аудио: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    logger.info(f"Транскрипция аудио...")
    try:
        result = model.transcribe(audio, batch_size=batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA out of memory во время транскрипции. Повторяем с batch_size=1...")
            torch.cuda.empty_cache()
            result = model.transcribe(audio, batch_size=1)
        else:
            raise e

    logger.info(f"Транскрипция завершена. Найдено сегментов: {len(result['segments'])}")

    language_code = result.get("language", "en")
    logger.info(f"Определён язык: {language_code}")

    if language_code not in ["en", "ru", "ro"]:
        logger.warning(f"Определённый язык ({language_code}) не входит в список поддерживаемых (en, ru, ro)!")

    gc.collect()
    torch.cuda.empty_cache()
    del model

    logger.info(f"Выравнивание сегментов по словам...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    logger.info(f"Выполняем диаризацию спикеров...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    try:
        diarize_segments = diarize_model(audio)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA out of memory во время диаризации. Повторяем на CPU...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
            diarize_segments = diarize_model(audio)
        else:
            raise e

    logger.info(f"Присвоение ID спикеров каждому слову...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    logger.info(f"Анализ аудио успешно завершён.")

    return {
        "segments": result["segments"],
        "diarization": diarize_segments,
        "language": language_code
    }


def save_audio_result(result: dict, output_path: str):
    """
    Сохраняет результат аудиоанализа в JSON файл.
    """

    logger.info(f"Сохранение результатов аудиоанализа в {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable_result = {
        "language": result["language"]
    }

    # Обработка сегментов
    segments = result["segments"]
    serializable_segments = []
    
    for seg in segments:
        # Преобразуем сегмент в словарь, если это не словарь
        if not isinstance(seg, dict):
            # Если это объект с атрибутами
            if hasattr(seg, "__dict__"):
                seg_dict = seg.__dict__.copy()
            else:
                # Если это объект другого типа, пробуем преобразовать его основные атрибуты
                seg_dict = {
                    "start": float(getattr(seg, "start", 0)),
                    "end": float(getattr(seg, "end", 0)),
                    "text": str(getattr(seg, "text", "")),
                }
        else:
            seg_dict = seg.copy()
        
        # Обработка слов, если они есть
        if "words" in seg_dict and seg_dict["words"]:
            clean_words = []
            for word in seg_dict["words"]:
                if not isinstance(word, dict):
                    if hasattr(word, "__dict__"):
                        word_dict = word.__dict__.copy()
                    else:
                        word_dict = {
                            "word": str(getattr(word, "word", "")),
                            "start": float(getattr(word, "start", 0)),
                            "end": float(getattr(word, "end", 0)),
                            "speaker": getattr(word, "speaker", None)
                        }
                else:
                    word_dict = word.copy()
                
                # Преобразование всех значений в сериализуемые типы
                for key in word_dict:
                    if not isinstance(word_dict[key], (str, int, float, bool, type(None))):
                        word_dict[key] = str(word_dict[key])
                
                clean_words.append(word_dict)
            seg_dict["words"] = clean_words
        
        # Преобразование всех значений сегмента в сериализуемые типы
        for key in list(seg_dict.keys()):
            if not isinstance(seg_dict[key], (str, int, float, bool, list, dict, type(None))):
                seg_dict[key] = str(seg_dict[key])
        
        serializable_segments.append(seg_dict)
    
    serializable_result["segments"] = serializable_segments

    # Обработка диаризации
    diarization = result["diarization"]
    serializable_diarization = []
    
    if hasattr(diarization, "to_dict"):
        diarization_list = diarization.to_dict(orient="records")
    elif isinstance(diarization, list):
        diarization_list = diarization
    elif hasattr(diarization, "itertracks"):  # pyannote.core.Annotation
        diarization_list = [
            {
                "start": segment.start,
                "end": segment.end,
                "speaker": label
            }
            for segment, _, label in diarization.itertracks(yield_label=True)
        ]
    else:
        # Если не удалось распознать формат, пытаемся итерировать
        diarization_list = []
        try:
            for item in diarization:
                if hasattr(item, "segment"):
                    diarization_list.append({
                        "start": float(item["segment"].start),
                        "end": float(item["segment"].end),
                        "speaker": item["label"]
                    })
                else:
                    # Общий случай
                    entry = {}
                    if hasattr(item, "__dict__"):
                        entry = item.__dict__.copy()
                    elif isinstance(item, dict):
                        entry = item.copy()
                    else:
                        logger.warning(f"Неизвестный тип диаризации: {type(item)}")
                        continue
                    
                    # Убедимся, что ключи start, end и speaker существуют
                    if "start" not in entry and hasattr(item, "start"):
                        entry["start"] = float(item.start)
                    if "end" not in entry and hasattr(item, "end"):
                        entry["end"] = float(item.end)
                    if "speaker" not in entry and hasattr(item, "speaker"):
                        entry["speaker"] = item.speaker
                    
                    # Преобразование в стандартные типы
                    for key in list(entry.keys()):
                        if not isinstance(entry[key], (str, int, float, bool, type(None))):
                            entry[key] = str(entry[key])
                    
                    diarization_list.append(entry)
        except Exception as e:
            logger.error(f"Ошибка при обработке диаризации: {e}")
            diarization_list = [{"error": "Не удалось сериализовать данные диаризации"}]
    
    # Обработка каждой записи для обеспечения сериализуемости
    for entry in diarization_list:
        serializable_entry = {}
        for key, value in entry.items():
            # Преобразование всех значений в сериализуемые типы
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable_entry[key] = value
            else:
                # Попытка преобразовать в float, если это временная метка
                if key in ["start", "end"]:
                    try:
                        serializable_entry[key] = float(value)
                    except (TypeError, ValueError):
                        serializable_entry[key] = str(value)
                else:
                    serializable_entry[key] = str(value)
        
        serializable_diarization.append(serializable_entry)
    
    serializable_result["diarization"] = serializable_diarization

    # Проверка сериализуемости перед сохранением
    try:
        json.dumps(serializable_result)
    except TypeError as e:
        logger.error(f"Ошибка сериализации: {e}")
        # Более жёсткая обработка - преобразование всех проблемных объектов в строки
        for key, value in serializable_result.items():
            if isinstance(value, (list, dict)):
                serializable_result[key] = json.loads(json.dumps(value, default=str))
    
    # Сохраняем
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)

    logger.info(f"Результаты успешно сохранены.")


def load_audio_result_from_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def plot_speaker_timeline(diarization_data: list, title="Speaker Timeline", figsize=(12, 2), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)

    speakers = sorted(set(d['speaker'] for d in diarization_data))
    speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

    for entry in diarization_data:
        start = entry['start']
        end = entry['end']
        speaker = entry['speaker']
        ax.hlines(y=speaker_to_y[speaker], xmin=start, xmax=end, linewidth=6)

    ax.set_yticks(list(speaker_to_y.values()))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Время (сек.)")
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"График сохранён в {save_path}")
    else:
        plt.tight_layout()
        plt.show()