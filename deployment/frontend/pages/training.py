# Страница обучения моделей
import streamlit as st
import requests
from deployment.frontend.utils.api_client import *

def show_page():
    st.header("Обучение модели")
    st.write("Настройте параметры модели и запустите процесс обучения")

    # Задаем параметр Альфа
    alpha = st.slider("Коэффициент регуляризации (alpha)", 0.01, 50.0, 1.0)

    # Задаем идентификатор модели
    model_id = st.text_input("Идентификатор модели", "model_1")

    # Выбор типа модели
    model_type = st.selectbox(
        "Тип модели",
        options=["full", "poly", "ohe"],
        index=0,
        help="Выберите тип модели для обучения"
    )

    # Обучение модели
    if st.button("Обучить модель"):
        if not model_id:
            st.error("Пожалуйста, укажите идентификатор модели.")
        else:
            with st.spinner("Обучение модели, пожалуйста подождите..."):
                response = train_model(model_id, model_type, alpha, "api/v1/models/fit")
            if response.status_code == 200:
                result = response.json()
                st.success(result['message'])
                st.write(f"Сообщение: {result['model']}")
            else:
                st.error(f"Ошибка: {response.json()['detail']}")

    # Список  готовых моделей
    st.subheader("Список доступных моделей")
    if st.button("Обновить список"):
        response = list_models("api/v1/models/list_models")
        if response.status_code == 200:
            list_model = response.json().get("models", [])
            st.success(response.json().get("models", []))
            st.write(f"{type(list_model)}")
            st.subheader("Кривая обучения")
            model_to_visualize = st.selectbox("Выберите модель для построения графика:", list_model)

            if st.button("Показать кривую обучения"):
                response = learning_curve("api/v1/models/learning_curve", model_to_visualize)
                if response.status_code == 200:
                    learning_curve_data = response.json()
                    st.write(f"Кривая обучения для модели '{model_to_visualize}':")
                    st.line_chart({
                        "Ошибка на трейне": learning_curve_data[1],
                        "Ошибка на тесте": learning_curve_data[2]
                    })
                else:
                    st.error(f"Ошибка: {response.json()['detail']}")
        else:
            st.error(f"Ошибка {response.status_code}: {response.text}")