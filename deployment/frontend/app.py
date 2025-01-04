# Основной файл Streamlit-приложения
import streamlit as st
from pages import eda, training, inference
# Настройки приложения
st.set_page_config(page_title="Car Valuation App", layout="wide")

# Заголовок
st.title("Car Valuation App")

# Меню
menu = st.sidebar.radio("Навигация", ["Анализ данных", "Обучение модели", "Инференс"])

# Переключение между страницами
if menu == "Анализ данных":
    eda.show_page()
elif menu == "Обучение модели":
    training.show_page()
elif menu == "Инференс":
    inference.show_page()