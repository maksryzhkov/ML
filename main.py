from src.eda import generate_eda_plots
from src.data_preparation import load_and_prepare_data
from src.models_training import train_all_models
from src.evaluation import evaluate_and_compare
from src.shap_analysis import generate_shap_analysis


def main():
    print("ПРОГНОЗИРОВАНИЕ СТОИМОСТИ КОМПАНИЙ")

    try:
        # 1. EDA
        generate_eda_plots()

        # 2. Подготовка данных
        data = load_and_prepare_data()

        # 3. Обучение моделей
        models = train_all_models(
            data["X_train"],
            data["X_test"],
            data["y_train_log"],
            data["y_test_log"]
        )

        # 4. Оценка и сравнение
        best_model_name = evaluate_and_compare(
            models,
            data["X_train"],
            data["X_test"],
            data["y_train_log"],
            data["y_test_real"],
            data["names_test"]
        )

        # 5. SHAP-анализ лучшей модели
        print(f"\nSHAP-анализ для лучшей модели: {best_model_name}")
        generate_shap_analysis(
            best_model_name=best_model_name,
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train_log=data["y_train_log"],
            y_test_real=data["y_test_real"],
            names_test=data["names_test"]
        )
        print("ВСЕ МОДЕЛИ СОХРАНЕНЫ, СРАВНЕНИЕ ПРОВЕДЕНО")

    except Exception as e:
        print(f"ОШИБКА ВЫПОЛНЕНИЯ: {e}")
        raise

if __name__ == "__main__":
    main()