# ML Ops: Docker-сервис для fraud detection

В этом проекте я упаковал модель для задачи детекции фрода карточных транзакций в Docker-контейнер.

Основная идея такая: есть файл `test.csv`, контейнер его читает, прогоняет через модель и сохраняет результат в формате `sample_submission.csv`.

Модель внутри контейнера не обучается. Контейнер нужен только для inference, то есть для получения предсказаний на новых данных.

## Что делает проект

Проект решает задачу из соревнования ML 2025 по определению мошеннических карточных транзакций.

На вход подаётся файл:

```text
input/test.csv
```

На выходе получается файл:

```text
output/sample_submission.csv
```

В выходном файле для каждой строки из `test.csv` записывается предсказание модели. Обычно это вероятность того, что транзакция является мошеннической.

## Структура проекта

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── train_model.py
├── sample_submission.csv
├── train.csv
├── test.csv
├── app/
│   └── app.py
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── score.py
│   ├── save_submission.py
│   └── pipeline.py
├── models/
│   └── .gitkeep
├── input/
│   └── .gitkeep
└── output/
    └── .gitkeep
```

Кратко по основным файлам:

* `train_model.py` — обучает модель и сохраняет её в `models/model.joblib`;
* `src/load_data.py` — загружает входной `test.csv`;
* `src/preprocess.py` — делает обработку признаков;
* `src/score.py` — загружает модель и получает предсказания;
* `src/save_submission.py` — сохраняет итоговый файл;
* `src/pipeline.py` — запускает весь inference по шагам;
* `Dockerfile` — описывает сборку Docker-образа;
* `requirements.txt` — зависимости Python.

## Зависимости

Для проекта используются основные библиотеки:

```text
joblib
numpy
pandas
scikit-learn
```

Они нужны для работы с таблицами, обработки признаков, обучения модели и сохранения модели в файл.

Установка зависимостей:

```bash
python -m pip install -r requirements.txt
```

Если на Linux команда `python` не работает, можно использовать:

```bash
python3 -m pip install -r requirements.txt
```

## Подготовка модели

Сначала нужно обучить модель локально:

```bash
python train_model.py
```

После этого должен появиться файл:

```text
models/model.joblib
```

Именно этот файл потом будет использоваться контейнером.

В контейнере модель не обучается, потому что по заданию нужен только inference. Поэтому перед сборкой Docker-образа файл `models/model.joblib` уже должен существовать.

## Как работает pipeline

При запуске контейнера выполняется такая последовательность:

```text
input/test.csv
    ↓
загрузка данных
    ↓
препроцессинг
    ↓
загрузка models/model.joblib
    ↓
получение предсказаний
    ↓
output/sample_submission.csv
```

То есть пользователь кладёт `test.csv` в папку `input`, запускает контейнер, а результат появляется в папке `output`.

## Сборка Docker image

Перед сборкой нужно убедиться, что модель уже создана:

```bash
ls models
```

Там должен быть файл:

```text
model.joblib
```

После этого можно собрать Docker image:

```bash
docker build -t fraud-mlops-service .
```

Если на Linux возникает ошибка с правами Docker, можно запустить через `sudo`:

```bash
sudo docker build -t fraud-mlops-service .
```

## Запуск контейнера на Linux/macOS

Создаём папки и кладём тестовый файл:

```bash
mkdir -p input output
cp test.csv input/test.csv
```

Запускаем контейнер:

```bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  fraud-mlops-service
```

Если Docker требует права администратора:

```bash
sudo docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  fraud-mlops-service
```

После запуска результат будет здесь:

```text
output/sample_submission.csv
```

## Запуск на Windows PowerShell

```powershell
mkdir input
mkdir output
copy test.csv input/test.csv

docker run --rm `
  -v ${PWD}/input:/app/input `
  -v ${PWD}/output:/app/output `
  fraud-mlops-service
```

## Проверка результата

Можно проверить, что файл появился:

```bash
ls -lh output
```

И посмотреть первые строки:

```bash
head output/sample_submission.csv
```

Пример результата:

```text
index,prediction
0,0.2149851375610224
1,0.3739590559927797
2,0.009669153036213815
```

Также можно проверить, что формат совпадает с шаблоном:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

sample_path = Path("sample_submission.csv")
if not sample_path.exists():
    sample_path = Path("sample_submition.csv")

sample = pd.read_csv(sample_path)
result = pd.read_csv("output/sample_submission.csv")

print("sample:", sample.shape)
print("result:", result.shape)

assert list(sample.columns) == list(result.columns), "Колонки не совпадают"
assert len(sample) == len(result), "Количество строк не совпадает"

print("OK")
PY
```

## Полный запуск с нуля

```bash
python -m pip install -r requirements.txt
python train_model.py

docker build -t fraud-mlops-service .

mkdir -p input output
cp test.csv input/test.csv

docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  fraud-mlops-service
```

После этого итоговый файл будет лежать здесь:

```text
output/sample_submission.csv
```

## Зачем здесь Docker

Docker нужен, чтобы проект запускался в одинаковом окружении. Без Docker у разных людей могут быть разные версии Python и библиотек, из-за чего код может не запуститься.

В Docker-образе заранее задаются:

* версия Python;
* зависимости из `requirements.txt`;
* код проекта;
* команда запуска inference.

Поэтому другой человек может просто собрать образ и запустить контейнер, не настраивая всё вручную.

## Что получилось на выходе

В результате получился простой batch inference сервис:

* вход: `input/test.csv`;
* модель: `models/model.joblib`;
* выход: `output/sample_submission.csv`;
* запуск: через Docker.

Основная цель работы — не добиться максимального качества модели, а показать, что ML-модель можно упаковать в воспроизводимый сервис и запустить через Docker.
