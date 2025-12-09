"""
Утилиты для проекта прогнозирования стоимости компаний
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, Any


def correct_predictions(
        model: Any,
        X_train: pd.DataFrame,
        y_train_log: pd.Series,
        X_test: pd.DataFrame
) -> Tuple[np.ndarray, float]:
    """
    Корректирует предсказания модели для компенсации систематической ошибки

    Parameters:
    -----------
    model : обученная модель
    X_train : признаки обучающей выборки
    y_train_log : логарифмированный таргет обучающей выборки
    X_test : признаки тестовой выборки

    Returns:
    --------
    test_pred_corrected : скорректированные предсказания
    correction_factor : коэффициент коррекции
    """
    train_pred = model.predict(X_train)
    correction = np.exp(y_train_log.mean() - train_pred.mean())
    test_pred_log = model.predict(X_test)
    test_pred_corrected = np.expm1(test_pred_log) * correction

    return test_pred_corrected, correction


def save_plot(
        fig: plt.Figure,
        filename: str,
        plots_dir: Path,
        dpi: int = 300
) -> None:
    """
    Сохраняет график с обработкой ошибок

    Parameters:
    -----------
    fig : объект Figure
    filename : имя файла
    plots_dir : директория для сохранения
    dpi : разрешение
    """
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        filepath = plots_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"- График сохранен: {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении графика {filename}: {e}")
        raise


def load_model(model_name: str, models_dir: Path) -> Any:
    """
    Загружает модель из файла с проверкой

    Parameters:
    -----------
    model_name : название модели
    models_dir : директория с моделями

    Returns:
    --------
    model : загруженная модель
    """
    model_mapping = {
        "Linear Regression": "linear_regression.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
    }

    if model_name not in model_mapping:
        available = ", ".join(model_mapping.keys())
        raise ValueError(
            f"Модель '{model_name}' не найдена. "
            f"Доступные модели: {available}"
        )

    model_path = models_dir / model_mapping[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    return joblib.load(model_path)


def safe_filename(text: str, max_length: int = 50) -> str:
    """
    Создает безопасное имя файла из строки

    Parameters:
    -----------
    text : исходный текст
    max_length : максимальная длина

    Returns:
    --------
    safe_name : безопасное имя файла
    """
    # Заменяем неалфавитно-цифровые символы на подчеркивания
    safe = "".join(c if c.isalnum() else "_" for c in str(text))
    # Убираем повторяющиеся подчеркивания
    safe = "_".join(filter(None, safe.split("_")))
    # Обрезаем до максимальной длины
    return safe[:max_length]