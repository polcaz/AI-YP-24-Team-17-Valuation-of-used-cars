# Страница для инференса
import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

def make_prediction(model_id, input_data):
    url = f"{API_BASE_URL}/predict/{model_id}"
    response = requests.post(url, json={"data": input_data})
    return response.json()

def show_page():
    st.header("Инференс")
    st.write("Сделайте предсказание для новых данных.")

    model_id = st.text_input("Идентификатор модели", "1")
    input_data = st.text_area("Введите данные в формате JSON", "{}")

    if st.button("Сделать предсказание"):
        try:
            response = make_prediction(model_id=model_id, input_data=input_data)
            st.write("Результат предсказания:", response)
        except Exception as e:
            st.error(f"Ошибка: {e}")