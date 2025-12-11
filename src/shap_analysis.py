# Делает SHAP-анализ для лучшей модели: график №4 + №5 для топ-3 ошибок

import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from .config import MODELS_DIR, PLOTS_DIR


def correct_predictions(model, X_train, y_train_log, X_test):
    """Универсальная функция коррекции предсказаний (вынесена из evaluation.py)"""
    train_pred = model.predict(X_train)
    correction = np.exp(y_train_log.mean() - train_pred.mean())
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log) * correction
    return test_pred, correction


def generate_shap_analysis(best_model_name, X_train, X_test, y_train_log, y_test_real, names_test):
    """
    Улучшенная версия SHAP-анализа без циклических импортов

    Parameters:
    -----------
    best_model_name : str
        Название лучшей модели
    X_train, X_test : DataFrame
        Признаки для обучения и теста
    y_train_log : Series
        Логарифмированные значения target для обучения
    y_test_real : Series
        Реальные значения target для теста
    names_test : Series
        Названия компаний в тестовой выборке
    """
    print("\nГенерация SHAP-анализа...")

    try:
        # Какой файл соответствует названию модели
        name_to_file = {
            "Linear Regression": "linear_regression.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
            "XGBoost": "xgboost.pkl",
            "LightGBM": "lightgbm.pkl",
        }

        if best_model_name not in name_to_file:
            raise ValueError(f"Модель {best_model_name} не найдена в доступных моделях")

        model_path = MODELS_DIR / name_to_file[best_model_name]
        model = joblib.load(model_path)
        print(f"- Загружена модель: {best_model_name}")

        # Коррекция предсказаний
        test_pred, _ = correct_predictions(model, X_train, y_train_log, X_test)

        # Находим самые большие ошибки
        errors = np.abs(y_test_real - test_pred)
        top_n = min(3, len(errors))
        top_pos = np.argsort(errors.values)[-top_n:][::-1]

        # Считаем SHAP-значения
        print("- Считаем SHAP...")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # График №04 — общая важность признаков
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, max_display=12, show=False)
        plt.title(f"SHAP-анализ: {best_model_name}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "4_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("- 4_shap_summary.png сохранен")

        # Графики №05 — почему модель ошиблась на конкретных компаниях
        for i, pos in enumerate(top_pos, 1):
            name = names_test.iloc[pos]
            real = y_test_real.iloc[pos] / 1e9
            pred = test_pred[pos] / 1e9
            error_pct = errors.iloc[pos] / y_test_real.iloc[pos] * 100

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    base_values=explainer.expected_value,
                    values=shap_values.values[pos],
                    data=X_test.iloc[pos],
                    feature_names=X_test.columns
                ),
                max_display=10,
                show=False
            )
            plt.title(
                f"{i}. {name}\n"
                f"Реальная стоимость: {real:.1f} млрд $\n"
                f"Прогноз: {pred:.1f} млрд $\n"
                f"Ошибка: {error_pct:.1f}%",
                fontsize=12
            )

            # Безопасное имя для файла
            safe_name = "".join(c if c.isalnum() else "_" for c in str(name))
            safe_name = safe_name[:50]  # Ограничиваем длину

            plt.savefig(
                PLOTS_DIR / f"5_shap_waterfall_{safe_name}_rank{i}.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()
            print(f"- График для {name} сохранен")

        print(f"- Всего сохранено {top_n} графиков водопада")
        print("SHAP-анализ успешно завершен\n")

    except FileNotFoundError as e:
        print(f"Ошибка: файл модели не найден: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка в SHAP-анализе: {e}")
        raise
