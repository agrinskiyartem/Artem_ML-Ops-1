# Docker-сервис для детекции фрода карточных транзакций

Проект упаковывает простую CPU-only ML-модель в Docker-сервис для batch inference. Контейнер читает файл `/app/input/test.csv`, применяет заранее обученную модель из `/app/models/model.joblib` и сохраняет результат в `/app/output/sample_submission.csv` в формате шаблона соревнования.

> Важно: контейнер **не обучает модель**. Перед сборкой Docker image нужно локально подготовить артефакт `models/model.joblib` командой `python train_model.py`.

## Структура репозитория

```text
.
├── Dockerfile
├── README.md
├── requirements.txt
├── train_model.py              # локальное обучение и сохранение models/model.joblib
├── sample_submition.csv        # шаблон submission из соревнования; поддерживается и имя sample_submission.csv
├── src/
│   ├── config.py               # пути, списки колонок и фичей
│   ├── load_data.py            # загрузка и проверка test.csv
│   ├── preprocess.py           # feature engineering для inference
│   ├── score.py                # загрузка модели и скоринг
│   ├── save_submission.py      # сохранение результата по шаблону
│   └── pipeline.py             # последовательный запуск всех этапов inference
├── app/
│   └── app.py                  # совместимый entrypoint, запускающий pipeline
├── models/
│   └── .gitkeep                # сама модель model.joblib не хранится в Git
├── input/
│   └── .gitkeep                # сюда пользователь кладёт test.csv
└── output/
    └── .gitkeep                # сюда контейнер пишет sample_submission.csv
```

## Входные и выходные файлы

### Вход

Контейнер ожидает файл:

```text
/app/input/test.csv
```

В локальном репозитории это обычно файл:

```text
input/test.csv
```

Обязательные колонки входного `test.csv`:

```text
transaction_time, merch, cat_id, amount, name_1, name_2, gender, street,
one_city, us_state, post_code, lat, lon, population_city, jobs,
merchant_lat, merchant_lon
```

Если каких-то обязательных колонок нет, сервис завершится с понятной ошибкой и списком пропущенных колонок. Лишние колонки допускаются и игнорируются. Колонка `target`, если случайно есть во входном файле, не используется при inference.

### Выход

Результат всегда сохраняется в:

```text
/app/output/sample_submission.csv
```

При запуске с volume этот файл появится локально как:

```text
output/sample_submission.csv
```

Колонки и порядок колонок берутся из шаблона `sample_submission.csv` или `sample_submition.csv`. Предсказания записываются в последнюю колонку шаблона, например `prediction`.

## Подготовка модели

В Git не добавляются бинарные файлы моделей. Перед сборкой Docker image нужно создать локальный артефакт:

```bash
python -m pip install -r requirements.txt
python train_model.py
```

Скрипт обучает простой `scikit-learn` pipeline:

- `transaction_time` преобразуется в час, день недели, месяц, день месяца и признак выходного дня;
- координаты клиента и продавца используются для расчёта расстояния `distance_km`;
- числовые признаки заполняются медианой и масштабируются;
- категориальные признаки заполняются значением `unknown` и кодируются через `OneHotEncoder`;
- classifier: `LogisticRegression` с `class_weight="balanced"`.

После успешного запуска должен появиться файл:

```text
models/model.joblib
```

Если `train.csv` в репозитории является только Git LFS pointer-файлом, сначала загрузите реальный датасет из соревнования или выполните `git lfs pull`, а затем повторите обучение.

## Сборка Docker image

Dockerfile ожидает, что `models/model.joblib` уже существует. Если файла нет, сборка завершится с понятным сообщением об ошибке.

### Linux/macOS

```bash
docker build -t fraud-mlops-service .
```

### Windows PowerShell

```powershell
docker build -t fraud-mlops-service .
```

## Запуск контейнера

### Linux/macOS

```bash
mkdir -p input output
cp test.csv input/test.csv
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  fraud-mlops-service
ls output
```

### Windows PowerShell

```powershell
mkdir input
mkdir output
copy test.csv input/test.csv
docker run --rm `
  -v ${PWD}/input:/app/input `
  -v ${PWD}/output:/app/output `
  fraud-mlops-service
dir output
```

## Проверка результата

Проверить, что файл создан и совпадает с шаблоном по количеству строк и колонкам, можно так:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

sample_path = Path('sample_submission.csv') if Path('sample_submission.csv').exists() else Path('sample_submition.csv')
submission = pd.read_csv('output/sample_submission.csv')
sample = pd.read_csv(sample_path)

assert list(submission.columns) == list(sample.columns), 'Колонки не совпадают с шаблоном'
assert len(submission) == len(sample), 'Количество строк не совпадает с шаблоном'
print('OK:', submission.shape)
PY
```

## Полный сценарий для Linux/macOS

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
ls output
```

## Полный сценарий для Windows PowerShell

```powershell
python -m pip install -r requirements.txt
python train_model.py

docker build -t fraud-mlops-service .
mkdir input
mkdir output
copy test.csv input/test.csv
docker run --rm `
  -v ${PWD}/input:/app/input `
  -v ${PWD}/output:/app/output `
  fraud-mlops-service
dir output
```

## Почему модель не хранится в Git

Файлы `models/*.joblib`, `models/*.pkl` и `models/*.cbm` игнорируются через `.gitignore`, чтобы Pull Request содержал только текстовые файлы. В репозитории остаётся `models/.gitkeep`, а модель создаётся локально перед сборкой Docker image.
