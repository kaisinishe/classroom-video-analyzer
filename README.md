# 📚 Проект: Анализатор видео школьных уроков и митингов

---

## Описание

AI-платформа для анализа видеозаписей уроков и рабочих встреч:
- Детектирование участников (ученики/учитель).
- Поведенческий анализ (позы, действия, эмоции).
- Аудио-анализ (транскрипция, диаризация по спикерам).
- Генерация структурированного отчета (JSON, PDF) и графиков активности.

---
# Клонируем проект
git clone https://github.com/kaisinishe/classroom-video-analyzer.git
cd classroom-video-analyzer

# Создаём виртуальное окружение
python -m venv venv

# Активируем его
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

---

video-analyzer/
├── audio_analysis/         # Аудио транскрипция, диаризация
├── preprocess/             # Препроцессинг видео
├── utils/                  # Утилиты (логирование и прочее)
├── assets/                 # Видео и аудио-файлы
├── notebooks/              # Jupyter ноутбуки для отладки
├── main.py                 # Основной скрипт запуска
├── requirements.txt        # Зависимости проекта
├── README.md               # Описание проекта
└── .gitignore              # Исключения Git

