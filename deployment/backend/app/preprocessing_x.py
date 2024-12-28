import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def preproc_x(df):

    # Класс для удаления столбцов из датасета
    class columnDropperTransformer():
        def __init__(self,columns):
            self.columns=columns
        def transform(self,X,y=None):
            X = X.drop(self.columns,axis=1)
            return X
        def fit(self, X, y=None):
            return self

    # Класс для обработки типа автомобиля
    class columnCarTypeTransformer():
        def __init__(self,columns):
            self.columns=columns
        def transform(self,X,y=None):
            def mod_car_type(x):
                car_type_short = [
                    'Внедорожник 3 дв.',
                    'Внедорожник 5 дв.',
                    'Внедорожник открытый',
                    'Кабриолет',
                    'Компактвэн',
                    'Купе',
                    'Лифтбек',
                    'Микровэн',
                    'Минивэн',
                    'Пикап Двойная кабина',
                    'Пикап Одинарная кабина',
                    'Пикап Полуторная кабина',
                    'Родстер',
                    'Седан',
                    'Спидстер',
                    'Тарга',
                    'Универсал 5 дв.',
                    'Фастбек',
                    'Фургон',
                    'Хэтчбек 3 дв.',
                    'Хэтчбек 4 дв.',
                    'Хэтчбек 5 дв.'
                ]
                for car_type in car_type_short:
                    if car_type in x:
                        x = car_type
                return x
            X['car_type'] = X['car_type'].apply(lambda x: mod_car_type(x) if not pd.isnull(x) else np.nan)
            return X
        def fit(self, X, y=None):
            return self

    # Класс для преобразования количества владельцев в число
    class classOwnsTransformer():

        def transform(self, X, y=None):
            X['count_owner'] = X['count_owner'].apply(lambda x: int(x.split()[0]) if not pd.isnull(x) else np.nan)
            return X

        def fit(self, X, y=None):
            return self

    # Готовим данные для обработки
    drop_columns = ['Unnamed: 0', 'car_price', 'car_model', 'car_gen', 'car_compl', 'url_car', 'ann_id', 'ann_date', 'ann_city',
                        'avail', 'original_pts', 'customs', 'link_cpl', 'eng_power_kw',
                        'pow_resrv', 'options', 'condition', 'url_compl', 'gross_weight']

    column_dropper= Pipeline([
        ("column_dropper", columnDropperTransformer(drop_columns))
    ])

    car_type_transformer = Pipeline([
        ("car_type_transformer", columnCarTypeTransformer('car_type'))
    ])

    class_owns_transformer = Pipeline([
        ("class_owns_transformer", classOwnsTransformer())
    ])

    df = column_dropper.fit_transform(df)
    df = car_type_transformer.fit_transform(df)
    df = class_owns_transformer.fit_transform(df)

    return df