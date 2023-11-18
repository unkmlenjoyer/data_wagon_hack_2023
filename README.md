# Data Wagon 2023 (hack)

# RU

### 1. Введение
-----

Целью проекта является создание системы предсказания отправления вагонов в плановый ремонт.

Решение нашей команды (`fit_predict`) заняло 8 место из более 50 команд на либерборде.

F1 = 50.55443 (публичный скор), 0.55730 (приватный срок)

### 2. Структура проекта
-----

- `data`: содержит данные для исследований
  - `final`: готовые данные, например, submissions
  - `processed`: обработанные данные
  - `raw`: содержит сырые данные
- `docs`: вспомогательная документация
  - `pdf`: презентация по хакатону
  - `images`: графики
- `research` содержит скрипты исследования (.py формата) и .ipynb:
  - `src`: готовые классы / утилиты для исследований
  - `data_generation.py`: скрипт генерации данных для обучения
  - `modeling.py`: скрипт моделирования и обучения Catboost
  - `predict_submission.py` скрипт создания сабмита для платформы
  - `main_notebook.ipynb` ноутбук, объединяющий в себе указанные выше 3 файла.

### 3. Установка
-----

Для работы использовался Python 3.11.0, необходимые библиотеки есть в `requirements.txt`


# EN

### 1. Intro
-----

The aim of project is development system for prediction railway cars repair time

Our team's solution (`fit_predict`) took 8th place out of more than 50 teams on the leaderboard.

F1 = 50.55443 (public score), 0.55730 (private score)

### 2. Project structure
-----

- `data`: сontains research data
  - `final`: submissions to leaderboard
  - `processed`: processed data ready for modeling
  - `raw`: raw data from many sources
- `docs`: documentation
  - `pdf`: hack presentation
  - `images`: graphs
- `research` contains research scripts in .py and .ipynb formats
  - `src`: useful utils
  - `data_generation.py`: data preprocessing
  - `modeling.py`: modeling by Catboost
  - `predict_submission.py` submit generator
  - `main_notebook.ipynb` notebook for orgs, just combination of 3 upper scripts.

### 3. Install
-----

We used Python 3.11.0, all dependencies are in `requrements.txt` file.