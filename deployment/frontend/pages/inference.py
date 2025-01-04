# Страница для инференса
import json

import streamlit as st
import requests
from deployment.frontend.utils.api_client import *

def show_page():
    st.header("Инференс")
    st.write("Сделайте предсказание для новых данных.")

    # Инициализация состояния
    if "list_model" not in st.session_state:
        st.session_state.list_model = []
    if "model_id" not in st.session_state:
        st.session_state.model_id = None

    # Кнопка для обновления списка моделей
    if st.button("Обновить список"):
        try:
            response = list_models("api/v1/models/list_models")
            if response and response.status_code == 200:
                st.session_state.list_model = response.json().get("models", [])
                if st.session_state.list_model:
                    st.success("Список моделей обновлен!")
                else:
                    st.warning("Модели не найдены.")
            else:
                st.error(f"Ошибка: сервер вернул код {response.status_code}")
        except Exception as e:
            st.error(f"Ошибка вызова API: {e}")

    # Выбор модели
    if st.session_state.list_model:
        st.session_state.model_id = st.selectbox(
            "Выберите модель для предсказания:",
            st.session_state.list_model,
            index=st.session_state.list_model.index(st.session_state.model_id)
            if st.session_state.model_id in st.session_state.list_model
            else 0
        )

    # Ввод данных для предсказания
    input_data = st.text_area("Введите данные в формате JSON", "{}")

    # Кнопка для выполнения предсказания
    if st.button("Сделать предсказание"):
        if not st.session_state.model_id:
            st.warning("Сначала выберите модель.")
        elif not input_data.strip():
            st.warning("Введите данные для предсказания.")
        else:
            try:
                # Отправка данных на сервер для предсказания
                # Преобразование строки JSON в словарь
                parsed_data = json.loads(input_data)
                response = make_prediction(
                    "api/v1/models/predict",
                    model_id=st.session_state.model_id,
                    input_data=parsed_data,
                )
                if response and response.status_code == 200:
                    predictions = response.json().get('predictions', [])
                    st.success("Предсказание выполнено успешно!")
                    st.write("Результат предсказания:", predictions)
                else:
                    st.error(f"Ошибка: сервер вернул код {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка выполнения предсказания: {e}")