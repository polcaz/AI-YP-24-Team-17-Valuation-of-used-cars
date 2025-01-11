import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PATH = '../data/'
MODEL_PATH = '../models/'


def get_cabin_type_and_compl(text):
    constructions = ['Внедорожник 3 дв.', 'Внедорожник 5 дв.', 'Внедорожник открытый', 'Кабриолет',
                     'Компактвэн', 'Купе', 'Купе-хардтоп', 'Лимузин', 'Лифтбек', 'Микровэн', 'Минивэн',
                     'Пикап Двойная кабина', 'Пикап Одинарная кабина', 'Пикап Полуторная кабина',
                     'Родстер', 'Седан', 'Седан 2 дв.', 'Седан-хардтоп', 'Спидстер', 'Тарга', 'Универсал 3 дв.',
                     'Универсал 5 дв.', 'Фастбек', 'Фургон', 'Хэтчбек 3 дв.', 'Хэтчбек 4 дв.',
                     'Хэтчбек 5 дв.']
    splitted_text = text.split()

    if len(splitted_text) >= 3:
        one_word = splitted_text[0]
        two_word = ' '.join(splitted_text[:2])
        three_word = ' '.join(splitted_text[:3])
        if three_word in constructions:
            if len(splitted_text) > 3:
                result = [three_word, ' '.join(splitted_text[3:])]
            else:
                result = [three_word, '']
        elif two_word in constructions:
            result = [two_word, ' '.join(splitted_text[2:])]
        elif one_word in constructions:
            result = [one_word, ' '.join(splitted_text[1:])]
        else:
            result = ['', ' '.join(splitted_text)]
    elif len(splitted_text) == 2:
        one_word = splitted_text[0]
        two_word = ' '.join(splitted_text[:2])
        if two_word in constructions:
            result = [two_word, '']
        elif one_word in constructions:
            result = [one_word, ' '.join(splitted_text[1:])]
        else:
            result = ['', ' '.join(splitted_text)]
    elif len(splitted_text) == 1:
        one_word = splitted_text[0]
        if one_word in constructions:
            result = [one_word, '']
        else:
            result = ['', ' '.join(splitted_text)]
    else:
        result = ''
    return result


# Определим трансформер, который из тестового столбца, в котором имеются
# как числа, так и диапазоны в виде 'n, m', 'n-m', или 'n/m
class MaxValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X_out = X.copy()
        self.features = X_out.columns

        # Создадим столбец с максимальным значением
        for col in self.features:
            if col == 'seat_count':
                X_out.loc[X_out['seat_count'].str.len() > 2, col + '_max'] = \
                    X_out[X_out['seat_count'].str.len() > 2]['seat_count'].str.split(', ').apply(
                    lambda x: max(map(int, x)))
            elif col == 'clearence':
                X_out.loc[X_out['clearence'].str.len() > 5, col + '_max'] = \
                    X_out[X_out['clearence'].str.len() > 5]['clearence'].str.split('-').apply(
                        lambda x: max(map(int, x)))
            else:
                X_out.loc[X_out['v_bag'].str.find('/') > 0, col + '_max'] = \
                    X_out[X_out['v_bag'].str.find('/') > 0]['v_bag'].str.split('/').apply(
                        lambda x: max(map(int, x)))
            # Пропуски заполним значениями из столбца с количеством сидений
            X_out.loc[X_out[col + '_max'].isna(), col + '_max'] = \
                X_out[X_out[col + '_max'].isna()][col].astype('float')

            # Заменим исходный столбец
            X_out[col] = X_out[col + '_max']
            X_out.drop([col + '_max'], axis=1, inplace=True)
        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования признака количества владельце
class CountOwnerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Заменим 1 на 1, 2 на 2, 3 и более - на 3
        X_out.loc[X_out['count_owner'] == '1 владелец', 'count_owner'] = 1
        X_out.loc[X_out['count_owner'] == '2 владельца', 'count_owner'] = 2
        X_out.loc[X_out['count_owner'] == '3 или более', 'count_owner'] = 3
        X_out.loc[X_out['count_owner'].isna(), 'count_owner'] = 3

        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования признака поколения авто
class CarGenTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Выделим обозначение поколения
        X_out['temp'] = X_out['car_gen'].str.extract(
            r'(^[IVX]{1}[IVX]*\b|\b[IVX]{1}[IVX]*\b|\b[IVX]{1}[IVX]*$|^\d+-\w+\.*\w*\.*$|^\d+-\w+\.*\w*\.*\b|\b\d+-\w+\.*\w*\.*\b|\b\d+-\w+\.*\w*\.*$)')
        # Создадим столбец с поколением
        X_out['generation'] = X_out['temp']
        X_out.loc[X_out['generation'].isna(), 'generation'] = ''

        # Выделим обозначение рестайлинга
        X_out['temp'] = X_out['car_gen'].str.extract(
            r'(^[Рр]ест\w*\s*\d*$|^[Рр]ест\w*\s*\d*\b|\b[Рр]ест\w*\s*\d*\b|\b[Рр]ест\w*\s*\d*$)')
        X_out.loc[X_out['temp'].isna(), 'temp'] = ''
        # Исправим опечатку в обозначении
        X_out.loc[X_out['temp'] == 'Ресталинг 2', 'temp'] = 'Рестайлинг 2'
        # Запишем рестайлинг с большой буквы
        X_out.loc[X_out['temp'] == 'рестайлинг', 'temp'] = 'Рестайлинг'
        X_out['car_gen'] = X_out['generation'] + ' ' + X_out['temp']
        X_out.loc[X_out['car_gen'] == '', 'car_gen'] = 'no_gen'
        X_out.drop(['generation', 'temp'], axis=1, inplace=True)

        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования типа кузова и комплектации
class CarTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Добавим к car_compl информацию о комплектации
        X_out['car_compl'] = X_out['car_compl'] + ' ' + \
             X_out['car_type'].apply(get_cabin_type_and_compl).apply(lambda x: x[1])

        # Тип кузова заменим на название конструкции кузова
        X_out['car_type'] = X_out['car_type'].apply(
            get_cabin_type_and_compl).apply(lambda x: x[0])

        # Уберём избыточную информацию из car_compl
        # Уберём выражение для мощности из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(r'( \(\d{1,4} л.с.\))', '', regex=True)
        # Уберём выражение для мощности в кВт из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'( \(\d{1,4}\.?\d* к[Вв]т\))', '', regex=True)
        # Создадим новый признак ёмкость батареи
        X_out['acc_capacity'] = X_out['car_compl'].str.extract(r'(\d+\.*\d*\s*k[Ww]h)')
        X_out['acc_capacity'] = X_out['acc_capacity'].str.replace('kWh', '').str.strip().astype(float)
        X_out.loc[X_out['acc_capacity'].isna(), 'acc_capacity'] = 0
        # Уберём выражение для ёмкости батареи car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(\d+\.*\d*\s*k[Ww]h)', '', regex=True)
        # Уберём выражение для объёма двигателя из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(^\d{1}\.\d{1}|\b\d{1}\.\d{1})', '', regex=True)
        # Уберём выражение для типа коробки из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(^MT|^AT|^AMT|^CVT|\bMT|\bAT|\bAMT|\bCVT)', '', regex=True)
        # Уберём выражение для типа привода из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(^4WD\b|\b4WD\b|\b4WD$|x[Dd]rive|\b4X\b|4x4)', '', regex=True)
        # Уберём выражение для типа двигателя из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(^d{2}|\bd{2}|[Ee]lectro|[Dd]iesel|hyb|dd$|dd\b|\bd\b|^b\b|^d$|\bd$)',
            '', regex=True)
        # Выделим обозначение турбины
        X_out['temp'] = X_out['car_compl'].str.extract(
            r'(TFSI|TDI|TSI|Diesel|TDId|Turbo|^T{1}$|^T{1}\b|\bT{1}$|\bT{1}\b)')
        # Заполним пропуски признака о турбине для турбированных моторов
        X_out.loc[~X_out['temp'].isna(), 'turbocharg'] = 'турбонаддув'
        # Уберём выражение для турбины из car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(TFSI|TDI|TSI|Diesel|TDId|Turbo|^T{1}$|^T{1}\b|\bT{1}$|\bT{1}\b)',
            '', regex=True)
        # Уберём выражение для cross car_compl
        X_out['car_compl'] = X_out['car_compl'].str.replace(
            r'(^Cross.*\b|\bCross.*\b|\b.*Cross\b|\b.*Cross$|^cross.*\b|\bcross.*\b|\b.*cross\b|\b.*cross$)',
            'Cross', regex=True)

        # Заменим несколько пробелов одним
        X_out['car_compl'] = X_out['car_compl'].str.replace(r'(\s+)', ' ', regex=True)

        # Заменим пустые строки заглушкой other
        X_out.loc[(X_out['car_compl'] == '') | (X_out['car_compl'] == ' '), 'car_compl'] = 'other'

        # Уберём одинаковые слова
        X_out['car_compl'] = X_out['car_compl'].str.split().apply(
            lambda x: ' '.join(list(set(x))) if len(x) > 1 else x[0])

        X_out.drop(['temp'], axis=1, inplace=True)

        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


class FeatureByAnotherFiller(BaseEstimator, TransformerMixin):
    """
    В датасете X в первом столбце имеются пропуски, которые
    заполняются на основе значений второго (и остальных).
    В столбцах, на основе которых заполняются пропуски, не должно
    быть пропусков. Значения числовых признаков кроме первого разбиваются
    на n_intervals интервалов c примерно одинаковым количеством объектов
    в каждом. Далее вычисляются медианные значения первой колонки с
    группировкой по значениям всех остальных признаков. У числовых признаков
    для группировки используются интервалы, которые были вычислены ранее.
    Далее пропущенные значения первого признака заполняются медианой (модой) по
    соответствующей группе.
    """

    def __init__(self, n_intervals: int = 10):
        self.n_intervals = n_intervals

    def fill_by_values(self, row):

        # Если категория не была в обучающей выборке, то возвращаем медиану
        # по всей обучающей выборке
        for col in self.cat_features:
            if row[col] not in self.values_to_fill.index.get_level_values(col):
                return self.avg_value_to_fill

        # Определяем интервалы, в которые попали числовые признаки
        # текущего объекта
        for col in self.num_features:
            if row[col] < self.intervals[col].min().left:
                row['interval'] = self.intervals[col].min()
                break
            if row[col] > self.intervals[col].max().right:
                row['interval'] = self.intervals[col].max()
                break
            for interval in self.intervals[col]:
                if row[col] in interval:
                    row['interval'] = interval
                    break
            row[col] = row['interval']
        # Если признаков для группировки более одного
        # то создадим мультииндекс
        if len(self.values_to_fill.index.names) > 1:
            result_index = []
            for col in self.values_to_fill.index.names:
                result_index.append(row[col])
            result_index = tuple(result_index)
            try:
                result = self.values_to_fill[result_index]
            except:
                result = self.avg_value_to_fill
        # Иначе просто возъмём значение группирующего признака в текущем
        # объекте для получения медианы
        else:
            result = self.values_to_fill[
                row[self.values_to_fill.index.names[0]]]
        return result

    def fit(self, X, y=None):
        data = X.copy()

        # Выделим признак для заполнения пропусков
        self.target = data.columns[0]

        # Создадим список числовых переменных для группировки
        self.num_features = data.iloc[:, 1:].select_dtypes(
            exclude=['object', 'category']).columns.to_list()

        # Составим список категориальных признаков для группировки
        self.cat_features = data.iloc[:, 1:].select_dtypes(
            include=['object', 'category']).columns.to_list()
        self.intervals = {}

        # Вычислим категории-интервалы для каждого числового признака
        for col in self.num_features:
            data[col] = pd.qcut(data[col], q=self.n_intervals)
            self.intervals[col] = data[col].unique()
        # Получим значение медианы (или моды) в зависимости от типа данных
        # целевого столбца
        if data[self.target].dtype in ['integer', 'floating']:
            self.values_to_fill = data.groupby(
                self.cat_features + self.num_features,
                observed=True)[self.target].agg('median')
            self.avg_value_to_fill = data[self.target].median()
        else:
            self.values_to_fill = data.groupby(
                self.cat_features + self.num_features,
                observed=True)[self.target].agg(
                    lambda x: None if x.mode().empty else x.mode().values[0]
            )
            self.avg_value_to_fill = data[self.target].mode().values[0]

        return self

    def transform(self, X):
        X_out = X.copy()
        self.features = X_out.columns
        X_out.loc[X_out[self.target].isna(), self.target] = \
            X_out[X_out[self.target].isna()].apply(self.fill_by_values, axis=1)

        # Все оставшиеся пропуски заполним общей медианой (модой) по обучающей
        # выборке
        X_out.loc[X_out[self.target].isna(), self.target] = \
        self.avg_value_to_fill

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования признака с редкими категориями
class RearValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.features = X.columns
        self.filters = {}
        for col in self.features:
            self.filters[col] = X[col].value_counts()[X[col].value_counts() < self.threshold]

        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.features:
            X_out.loc[X_out[col].isin(self.filters[col]), col] = 'Другие'
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования признаков связанных с типом
# двигателя и топлива
class EngTypeFuelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X_out = X.copy()
        self.features = X_out.columns
        # Для электромобилей заполним нулями
        X_out.loc[X_out['eng_type'] == 'Электро', 'v_tank'] = 0
        X_out.loc[X_out['eng_type'] == 'Электро', 'fuel_cons'] = 0
        # Для электромобилей пропуски заполним значением
        # ЭЭ (электроэнергия)
        X_out.loc[X_out['eng_type'] == 'Электро', 'fuel_brand'] = 'ЭЭ'
        # Для объектов с дизельным двигателем заполним пропуски значением Дизель
        X_out.loc[X_out['eng_type'] == 'Дизель', 'fuel_brand'] = 'ДТ'
        # Для объектов с двигателем на газу заполним пропуски значением Газ (Бензин)
        X_out.loc[X_out['eng_type'] == 'Газ', 'fuel_brand'] = 'Газ (Бензин)'

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для преобразования признака тип задних тормозов
class RearBrakesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        self.features = X_out.columns
        # Для барабанных передних тормозов заполним пропуски задних тормозов
        # барабанными.
        X_out.loc[X_out['front_brakes'] == 'барабанные', 'rear_brakes'] = 'барабанные'

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out.loc[(X_out['ann_date'] > '2024-11-28')&
                  (X_out['ann_date'] <= '2024-12-31'), 'ann_date'] = pd.to_datetime(
         X_out[(X_out['ann_date'] > '2024-11-28')&
                  (X_out['ann_date'] <= '2024-12-31')]['ann_date'].dt.strftime(
                date_format='%Y-%m-%d').replace(r'2024', '2023', regex=True), format='%Y-%m-%d')
        X_out['age'] = (X_out['ann_date'].dt.year - X_out.year).astype(int)
        X_out['age_mod'] = X_out['age'].apply(lambda x: np.log10(x) if x > 31 else x ** 0.25)

        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class ChangeScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, coef=1000):
        self.coef = coef

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Избавимся от лишних нулей
        X_out = X_out / self.coef

        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class MileageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Создадим искусственный признак
        X_out['mileage_mod'] = X_out['mileage'].apply(
            lambda x: np.log10(x) if x > 400 else x ** 0.25) ** 0.5
        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class AccelerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Создадим искусственный признак
        X_out['acceleration_mod'] = X_out['acceleration'].apply(
            lambda x: 1.5 * np.log10(x) if x > 20 else x ** 0.5)
        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class CylCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Создадим искусственный признак
        X_out['cyl_count_mod'] = X_out['cyl_count'].apply(
            lambda x: x ** 2 if x > 5 else 0.2 * x)
        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для добавления столбца с возрастом автомобиля
class StripTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Уберём лишние пробелы в начале и конце
        for col in X_out.columns:
            X_out[col] = X_out[col].str.strip()

        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Трансформер для заполнения пропусков нулями
class FillZeroTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        X_out = X_out.fillna(0).astype(float)

        self.features = X_out.columns

        return X_out

    def get_feature_names_out(self, X=None):
        return self.features

# Определим трансформер, который приводит к целочисленному типу
# передаваемые столбцы
class TypesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.features = X.columns
        return self

    def transform(self, X):
        X_out = X.copy()
        # # Приведём типы в соответствие с содержимым
        # X_out['pow_resrv'] = X_out['pow_resrv'].astype(float)
        X_out = X_out.astype({
            'car_make': 'category',
            'car_model': 'category',
            'car_gen': 'category',
            'car_type': 'category',
            'car_compl': 'object',
            #'ann_date': datetime64[ns],
            'car_price': float,
            'ann_city': 'category',
            'avail': 'category',
            'year': int,
            'mileage': float,
            'color': 'category',
            'eng_size': float,
            'eng_power': int,
            'eng_type': 'category',
            'pow_resrv': int,
            'options': 'object',
            'transmission': 'category',
            'drive': 'category',
            'st_wheel': 'category',
            'count_owner': int,
            'original_pts': 'category',
            'state_mark': 'category',
            'class_auto': 'category',
            'door_count': int,
            'long': int,
            'width': int,
            'height': int,
            'clearence': int,
            'v_tank': int,
            'curb_weight': int,
            'front_brakes': 'category',
            'rear_brakes': 'category',
            'max_speed': int,
            'acceleration': float,
            'fuel_cons': float,
            'fuel_brand': 'category',
            'engine_loc1': 'category',
            'engine_loc2': 'category',
            'turbocharg': 'category',
            'max_torq': int,
            'cyl_count': int,
            'seat_count': int,
            'v_bag': int,
            'acc_capacity': int,
            'age': int,
            'age_mod': float,
            'mileage_mod': float,
            'acceleration_mod': float,
            'cyl_count_mod': float
        })

        # Уменьшим избыточную разрядность чисел
        fcols = X_out.select_dtypes('float').columns
        icols = X_out.select_dtypes('integer').columns
        X_out[fcols] = X_out[fcols].apply(pd.to_numeric, downcast='float')
        X_out[icols] = X_out[icols].apply(pd.to_numeric, downcast='integer')
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features
