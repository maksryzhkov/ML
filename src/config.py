# Настройки проекта

from pathlib import Path

# Путь к корню проекта
ROOT_DIR = Path(__file__).parent.parent

# Пути к файлам и папкам
DATA_PATH = ROOT_DIR / "data" / "financials.csv"           # исходные данные
MODELS_DIR = ROOT_DIR / "models"                          # сюда сохраняем модели
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"                         # графики
LOGS_DIR = ROOT_DIR / "logs"                              # логи

# Создаём все нужные папки, если их нет
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Общие параметры
RANDOM_STATE = 42          # для воспроизводимости
TEST_SIZE = 0.2            # 20% данных — на тест
TARGET = "Market Cap"      # целевая переменная — рыночная капитализация

# Параметры моделей
MODEL_PARAMS = {
    "gradient_boosting": {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "min_samples_leaf": 3,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE
    },
    "lightgbm": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 7,
        "num_leaves": 64,
        "min_child_samples": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "verbose": -1
    }
}