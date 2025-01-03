import streamlit as st
import pandas as pd
import plotly.express as px
import io
import plotly.graph_objs as go
from deployment.frontend.utils.api_client import *

def show_page():
    st.header("Анализ данных")
    uploaded_file = st.file_uploader("Загрузите датасет", type=["csv"])
    train = None

    if uploaded_file is not None:

        dtypes_of_data = {
            'url_car': str,
            'car_make': str,
            'car_model': str,
            'car_gen': str,
            'car_type': str,
            'car_compl': str,
            'ann_id': str,
            'car_price': float,
            'ann_city': str,
            'link_cpl': str,
            'avail': str,
            'year': int,
            'mileage': int,
            'color': str,
            'eng_size': float,
            'eng_power': float,
            'eng_power_kw': float,
            'eng_type': str,
            'pow_resrv': str,
            'options': str,
            'transmission': str,
            'drive': str,
            'st_wheel': str,
            'condition': str,
            'count_owner': str,
            'original_pts': str,
            'customs': str,
            'url_compl': str,
            'state_mark': str,
            'class_auto': str,
            'door_count': float,  # могут быть пропуски
            'seat_count': str,  # могут быть значения с диапазоном, пропуски
            'long': float,
            'width': float,
            'height': float,
            'clearence': str,  # могут быть значения с диапазоном
            'v_bag': str,  # могут быть значения с диапазоном, пропуски
            'v_tank': float,
            'curb_weight': float,
            'gross_weight': float,
            'front_brakes': str,
            'rear_brakes': str,
            'max_speed': float,
            'acceleration': float,
            'fuel_cons': float,
            'fuel_brand': str,
            'engine_loc1': str,
            'engine_loc2': str,
            'turbocharg': str,
            'max_torq': float,
            'cyl_count': float  # Могут быть пропуски
        }
        df = pd.read_csv(uploaded_file, dtype=dtypes_of_data)
        df = df.rename(columns={'Unnamed: 0': 'Unnamed'})
        st.write("Просмотр данных:")
        st.dataframe(df)

        st.write("Общая информация о полях набора данных:")
        df_types = pd.DataFrame(df.dtypes)
        df_nulls = df.count()

        df_null_count = pd.concat([df_nulls, df_types], axis=1)
        df_null_count = df_null_count.reset_index()

        # Переименуем поля
        col_names = ["features", "non_null_counts", "types"]
        df_null_count.columns = col_names

        st.write(df_null_count)

        st.write("Основные характеристики числовых признаков:")
        st.write(df.describe(include=(float, int)))

        st.write("Основные характеристики нечисловых признаков:")
        st.write(df.describe(include=object))

        st.write("Распределение признака/целевой переменной:")
        column = st.selectbox("Выберите признак/целевую переменную", df.columns)
        st.bar_chart(df[column].value_counts())

        # Отправка файла в API
        st.header("Отправить файл в API")
        if st.button("Отправить"):
            response = upload_file("api/v1/dataset/upload", uploaded_file)
            if response.status_code == 200:
                result = response.json()
                st.success(result['message'])
            else:
                st.error('Ошибка отправки файла на сервер.')

        # Предобработка датасета в API
        st.header("Обработать датасет в API")
        if st.button("Обработать"):
            with st.spinner("Предобработка данных, пожалуйста подождите..."):
                response = preprocess_data("api/v1/dataset/preprocessing")
            if response.status_code == 200:
                result = response.json()
                st.success(result['message'])
                train = pd.DataFrame(result['train'])
            else:
                st.error('Ошибка обработки файла на сервере.')

    if train is not None:
        st.write("Просмотр обработанных данных:")
        st.dataframe(train)