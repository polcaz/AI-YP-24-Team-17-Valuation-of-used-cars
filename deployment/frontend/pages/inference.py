# Страница для инференса
import json
import pandas as pd
import streamlit as st
from deployment.frontend.utils.api_client import *


def show_page():
    st.header("Инференс")
    st.write("Сделайте предсказание для новых данных.")

    # Инициализация состояния
    if "list_model" not in st.session_state:
        st.session_state.list_model = []
    if "model_id" not in st.session_state:
        st.session_state.model_id = None
    if "train" not in st.session_state:
        st.session_state.train = None

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
            else 0,
        )

    # Проверка наличия загруженных данных
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        df = df.rename(columns={'Unnamed': 'Unnamed: 0'})
        categorical_columns = df.select_dtypes(include="object").columns
        numerical_columns = df.select_dtypes(include=["float", "int"]).columns

        st.subheader("Заполните значения для предсказания:")
        st.session_state.user_input = {}

        # Функция для фильтрации значений по выбранным ранее
        def filter_options(column):
            """Фильтрует значения текущего столбца на основе предыдущих выборов."""
            filtered_df = df.copy()
            for col, val in st.session_state.user_input.items():
                if val is not None and col in categorical_columns:
                    filtered_df = filtered_df[filtered_df[col] == val]
            return [None] + filtered_df[column].dropna().unique().tolist()

        # Динамическое заполнение данных
        for col in df.columns:
            if col in categorical_columns:
                options = filter_options(col)
                st.session_state.user_input[col] = st.selectbox(
                    f"{col} (категориальный)", options, key=col
                )
            elif col in numerical_columns:
                st.session_state.user_input[col] = st.number_input(
                    f"{col} (числовой)", value=None, key=col
                )
            else:
                st.session_state.user_input[col] = None

        # Кнопка для выполнения предсказания
        if st.button("Сделать предсказание"):
            st.write(st.session_state.user_input)
            if not st.session_state.model_id:
                st.warning("Сначала выберите модель.")
            else:
                try:
                    # Отправка данных на сервер для предсказания
                    response = make_prediction(
                        "api/v1/models/predict",
                        model_id=st.session_state.model_id,
                        input_data=st.session_state.user_input,
                    )
                    if response and response.status_code == 200:
                        predictions = response.json().get("predictions", [])
                        st.success("Предсказание выполнено успешно!")
                        st.write("Результат предсказания:", predictions)
                    else:
                        st.error(f"Ошибка: сервер вернул код {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка выполнения предсказания: {e}")
    else:
        st.warning("Пожалуйста, загрузите и обработайте данные на странице загрузки.")