# Сравнивает модели, делает таблицу и график №03, возвращает имя лучшей

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from .config import RESULTS_DIR, PLOTS_DIR


def correct_predictions(model, X_train, y_train_log, X_test):
    """Универсальная функция коррекции предсказаний"""
    train_pred = model.predict(X_train)
    correction = np.exp(y_train_log.mean() - train_pred.mean())
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log) * correction
    return test_pred, correction


def evaluate_and_compare(models, X_train, X_test, y_train_log, y_test_real, names_test):
    results = []
    print("Оценка моделей на тесте...")

    for name, model in models.items():
        try:
            # Используем общую функцию коррекции
            test_pred, _ = correct_predictions(model, X_train, y_train_log, X_test)

            # Считаем метрики качества
            r2 = r2_score(y_test_real, test_pred)
            mae = mean_absolute_error(y_test_real, test_pred) / 1e9

            results.append({
                "Модель": name,
                "R²": round(r2, 4),
                "MAE (млрд $)": round(mae, 2)
            })

        except Exception as e:
            print(f"Ошибка при оценке модели {name}: {e}")
            results.append({
                "Модель": name,
                "R²": 0.0,
                "MAE (млрд $)": float('inf')
            })

    # Сохраняем таблицу с результатами
    comparison = pd.DataFrame(results)
    comparison.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # График №3 — сравнение всех моделей
    plt.figure(figsize=(12, 7))
    x = np.arange(len(comparison))

    # Два столбца для R² и MAE
    plt.bar(
        x - 0.2, comparison["R²"],
        width=0.4, label="R²",
        color="skyblue", edgecolor="black"
    )
    plt.bar(
        x + 0.2, comparison["MAE (млрд $)"],
        width=0.4, label="MAE (млрд $)",
        color="lightcoral", edgecolor="black"
    )

    # Настройки графика
    plt.xticks(x, comparison["Модель"], rotation=15, ha="right")
    plt.title("Сравнение моделей: R² и MAE", fontsize=14, pad=20)
    plt.ylabel("Значение метрики")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Добавляем значения на столбцы
    for i, (r2, mae) in enumerate(zip(comparison["R²"], comparison["MAE (млрд $)"])):
        plt.text(i - 0.2, r2 + 0.01, f"{r2:.3f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + 0.2, mae + 0.01, f"{mae:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Выбираем лучшую модель
    best = comparison.loc[comparison["R²"].idxmax(), "Модель"]
    best_r2 = comparison.loc[comparison["R²"].idxmax(), "R²"]
    print(f"Лучшая модель по R²: {best} (R² = {best_r2})")

    return best