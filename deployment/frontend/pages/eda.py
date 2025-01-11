import streamlit as st
import pandas as pd
import plotly.express as px
import io
import plotly.graph_objs as go
from deployment.frontend.utils.api_client import *

def show_page():
    st.header("Анализ данных")
    uploaded_file = st.file_uploader("Загрузите датасет", type=["csv"])
    st.session_state.train = None
    st.session_state.df = None

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
        st.session_state.df = pd.read_csv(uploaded_file, dtype=dtypes_of_data)
        st.session_state.df = st.session_state.df.rename(columns={'Unnamed: 0': 'Unnamed'})
        st.write("Просмотр данных:")
        st.dataframe(st.session_state.df)

        st.write("Общая информация о полях набора данных:")
        df_types = pd.DataFrame(st.session_state.df.dtypes)
        df_nulls = st.session_state.df.count()

        df_null_count = pd.concat([df_nulls, df_types], axis=1)
        df_null_count = df_null_count.reset_index()

        # Переименуем поля
        col_names = ["features", "non_null_counts", "types"]
        df_null_count.columns = col_names

        st.write(df_null_count)

        st.write("Основные характеристики числовых признаков:")
        st.write(st.session_state.df.describe(include=(float, int)))

        st.write("Основные характеристики нечисловых признаков:")
        st.write(st.session_state.df.describe(include=object))

        st.write("Распределение признака/целевой переменной:")
        column = st.selectbox("Выберите признак/целевую переменную", st.session_state.df.columns)
        st.bar_chart(st.session_state.df[column].value_counts())

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
                st.session_state.train = pd.DataFrame(result['train'])
            else:
                st.error('Ошибка обработки файла на сервере.')

    if st.session_state.train is not None:
        st.write("Просмотр обработанных данных:")
        st.dataframe(st.session_state.train)

        # Составим список числовых признаков
        num_features = (
            st.session_state.train.select_dtypes(
                include=['int', 'float']
            ).columns.to_list()
        )

        message, select = st.columns(2, vertical_alignment="bottom")
        message.markdown('Выберите признак, распределение значений которого хотите изучить')

        feature_to_analize = select.selectbox('Признак', num_features)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
                x=st.session_state.train[feature_to_analize],
                histnorm='percent',
                name=f'{feature_to_analize}',
                marker_color='#EB89B5'
            ))

        fig.update_layout(
            xaxis_title_text=f'Значения {feature_to_analize}',
            yaxis_title_text='Количество объектов',
            title_text=f'Распределение значений {feature_to_analize}',
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Определим значения признака, для которых будем считать среднее
        st.session_state.train['intervals'], thresholds = \
            pd.qcut(st.session_state.train[feature_to_analize], q=40, duplicates='drop', retbins=True)
        data = st.session_state.train.groupby(by=['intervals'],
                          observed=True)['car_price'].agg(['mean'])
        labels = (thresholds[1:] + thresholds[:-1]) / 2

        fig = go.Figure([
            go.Scatter(
                name='объекты',
                x=st.session_state.train[feature_to_analize],
                y=st.session_state.train['car_price'],
                mode='markers',
                line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='среднее значение',
                x=labels,
                y=data['mean'],
                mode='markers',
                line=dict(color='red'),
            )
        ])
        fig.update_layout(
            yaxis=dict(title=dict(text='Целевой признак (цена)')),
            title=dict(text=f'Значения {feature_to_analize}'),
            title_text=f'Связь {feature_to_analize} с целевым признаком (цена)',
            hovermode="x"
        )
        st.plotly_chart(fig, use_container_width=True)