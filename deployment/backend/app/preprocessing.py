import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

def preproc(df):

    # Разделяем ИСХОДНЫЕ ДАННЫЕ на тренировочную и тестовую выборки
    df_train, df_test = train_test_split(df, train_size=0.75)

    # Класс для удаления столбцов из датасета
    class columnDropperTransformer():
        def __init__(self, columns):
            self.columns = columns

        def transform(self, X, y=None):
            X = X.drop(self.columns, axis=1)
            return X

        def fit(self, X, y=None):
            return self

    # Класс для обработки типа автомобиля
    class columnCarTypeTransformer():
        def __init__(self, columns):
            self.columns = columns

        def transform(self, X, y=None):
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

    # Создание класса-трансформера для заполнения пропусков медианой по группам
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    # Класс-трансформер для заполнения пропусков медианой по группам
    class GroupMedianImputer(BaseEstimator, TransformerMixin):
        def __init__(self, group_cols, target_col):
            """
            Инициализация трансформера.

            :param group_cols: Список колонок для группировки.
            :param target_col: Колонка с пропусками для заполнения.
            """
            self.group_cols = group_cols
            self.target_col = target_col
            self.group_medians_ = None

        def fit(self, X, y=None):
            # Рассчитываем медианы по комбинации значений в group_cols
            self.group_medians_ = (
                X.groupby(self.group_cols)[self.target_col]
                .median()
                .reset_index()
                .rename(columns={self.target_col: "median_value"})
            )
            return self

        def transform(self, X):
            X = X.copy()
            # Объединяем с рассчитанными медианами по ключевым столбцам
            X = X.merge(self.group_medians_, on=self.group_cols, how="left")
            # Заполняем пропуски в целевой колонке
            X[self.target_col] = X[self.target_col].fillna(X["median_value"])
            # Удаляем временную колонку с медианой
            X.drop(columns=["median_value"], inplace=True)
            return X

    # Создание класса-трансформера для заполнения пропусков модой по группам
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    # Класс-трансформер для заполнения пропусков модой по группам
    class GroupModeImputer(BaseEstimator, TransformerMixin):
        def __init__(self, group_cols, target_col):
            """
            Инициализация трансформера.

            :param group_cols: Список колонок для группировки.
            :param target_col: Колонка с пропусками для заполнения.
            """
            self.group_cols = group_cols
            self.target_col = target_col
            self.group_modes_ = None

        def fit(self, X, y=None):
            # Рассчитываем моды по комбинации значений в group_cols
            def safe_mode(series):
                if series.empty:
                    return np.nan
                mode = series.mode()
                return mode.iloc[0] if not mode.empty else np.nan

            self.group_modes_ = (
                X.groupby(self.group_cols)[self.target_col]
                .agg(safe_mode)
                .reset_index()
                .rename(columns={self.target_col: "mode_value"})
            )
            return self

        def transform(self, X):
            X = X.copy()
            # Объединяем с рассчитанными модами по ключевым столбцам
            X = X.merge(self.group_modes_, on=self.group_cols, how="left")
            # Заполняем пропуски в целевой колонке
            X[self.target_col] = X[self.target_col].fillna(X["mode_value"])
            # Удаляем временную колонку с модой
            X.drop(columns=["mode_value"], inplace=True)
            return X

    # Создание класса-трансформера для замены пропусков в столбце на основе условий из другого столбца
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    # Класс-трансформер для замены пропусков в столбце на основе условий из другого столбца
    class ConditionalValueImputer(BaseEstimator, TransformerMixin):
        def __init__(self, condition_col, target_col, fill_mapping):
            """
            Инициализация трансформера.

            :param condition_col: Колонка с условиями.
            :param target_col: Колонка, в которой будут заменяться пропуски.
            :param fill_mapping: Словарь, где ключ - значение из condition_col, а значение - значение для замены пропусков в target_col.
            """
            self.condition_col = condition_col
            self.target_col = target_col
            self.fill_mapping = fill_mapping

        def fit(self, X, y=None):
            return self  # Нет необходимости обучать

        def transform(self, X):
            X = X.copy()
            for condition_value, fill_value in self.fill_mapping.items():
                mask = (X[self.condition_col] == condition_value) & (X[self.target_col].isna())
                X.loc[mask, self.target_col] = fill_value
            return X

    # Создание класса-трансформера для преобразования строк, начинающихся с числа, в число
    import pandas as pd
    import numpy as np

    # Класс-трансформер для преобразования строк, начинающихся с числа, в число
    class NumberExtractorTransformer():
        def __init__(self, columns):
            """
            Инициализация трансформера.

            :param columns: Список названий столбцов, для которых нужно извлекать числа.
            """
            self.columns = columns

        def transform(self, X, y=None):
            """
            Извлекает числа из начала строки для указанных столбцов.

            :param X: Входной DataFrame.
            :param y: Не используется, добавлен для совместимости с API sklearn.
            :return: Преобразованный DataFrame.
            """
            X = X.copy()
            for column in self.columns:
                X[column] = (
                    X[column]
                    .astype(str)  # Преобразование в строку
                    .str.extract(r'^(\d+)')  # Извлечение первого числа в начале строки
                    .astype(float)  # Преобразование к числовому типу
                )
            return X

        def fit(self, X, y=None):
            return self  # Нет необходимости обучать

    # Класс для удаления дубликатов
    class DropDuplicate:
        def __init__(self, columns_to_exclude=None):
            """
            Класс для удаления дубликатов на основе выбранных столбцов.

            :param columns_to_exclude: Список столбцов, которые нужно исключить из проверки на дубликаты.
            """
            self.columns_to_exclude = columns_to_exclude or []

        def transform(self, X, y=None):
            """
            Удаляет дубликаты из DataFrame на основе заданных столбцов.

            :param X: Входной DataFrame.
            :param y: Не используется, добавлен для совместимости с API sklearn.
            :return: DataFrame без дубликатов.
            """
            X = X.copy()
            # Определение столбцов для проверки на дубликаты
            column_features = [col for col in X.columns if col not in self.columns_to_exclude]
            # Удаление дубликатов
            X = X.drop_duplicates(subset=column_features, keep='first').reset_index(drop=True)
            return X

        def fit(self, X, y=None):
            return self  # Нет необходимости обучать

    # Класс для обновления индекса
    class resetIndex():

        def transform(self, X, y=None):
            """
            Класс для обновления индекса.

            :param X: Входной DataFrame.
            :param y: Не используется, добавлен для совместимости с API sklearn.
            :return: DataFrame с обновленным индексом.
            """
            X = X.reset_index(drop=True)
            return X

        def fit(self, X, y=None):
            return self  # Нет необходимости обучать

    # Пока пустой пайплайн для обработки полей
    pipelines = []

    # Добавляем пайплайн для удаления полей, преобразования 'car_type' и
    # 'count_owner', 'seat_count', 'clearence', 'v_bag', 'fuel_cons'
    drop_columns = ['Unnamed: 0',
                    'url_car',
                    'ann_id',
                    'ann_date',
                    'ann_city',
                    'avail',
                    'original_pts',
                    'customs',
                    'link_cpl',
                    'eng_power_kw',
                    'pow_resrv',
                    'options',
                    'condition',
                    'url_compl',
                    'gross_weight']

    pipelines.append(("column_dropper", columnDropperTransformer(drop_columns)))
    pipelines.append(("car_type_transformer", columnCarTypeTransformer('car_type')))
    pipelines.append(("count_owner_transformer", NumberExtractorTransformer(
        columns=['count_owner', 'seat_count', 'clearence', 'v_bag', 'fuel_cons'])))

    # Добавляем пайплайн для замены пропусков в классе автомобиля ("class_auto")
    pipelines.append((f'class_auto_transformer_car_type', ConditionalValueImputer(
        condition_col='car_type',
        target_col='class_auto',
        fill_mapping={
            'Фургон': 'M',  # Для фургонов заполняем 'M'
            'Лимузин': 'F',  # Для лимузинов заполняем 'F'
        }
    )),
                     )

    strategy = [['car_make', 'car_model', 'car_gen', 'eng_type'],
                ['car_make', 'car_model', 'car_gen'],
                ['car_make', 'car_model', 'eng_type'],
                ['car_make', 'car_model'],
                ['car_type', 'eng_type'],
                ['car_type']
                ]

    target_columns = ['class_auto']

    for target_column in target_columns:
        for columns in strategy:
            pipelines.append(((f'group_mean_imputer_{target_column}_{columns}',
                               GroupModeImputer(group_cols=columns, target_col=target_column))))

    # Добавляем пайплайн для замены пропусков в числовых полях для электрокаров
    target_columns = ['eng_size', 'v_tank', 'fuel_cons', 'cyl_count']

    for target_column in target_columns:
        pipelines.append((f'eng_type_elektro_transformer_{target_column}', ConditionalValueImputer(
            condition_col='eng_type',
            target_col=target_column,
            fill_mapping={
                'Электро': 0,  # Для электрокаров заполняем 0
            }
        )),
                         )

    # Добавляем пайплайн для замены пропусков в текстовых полях для электрокаров
    target_columns = ['fuel_brand', 'engine_loc1', 'engine_loc2', 'turbocharg']

    for target_column in target_columns:
        pipelines.append((f'eng_type_elektro_transformer_{target_column}', ConditionalValueImputer(
            condition_col='eng_type',
            target_col=target_column,
            fill_mapping={
                'Электро': 'Nan',  # Для электрокаров заполняем 'Nan'
                }
            )),
        )

    # Добавляем пайплайн для замены пропусков в числовых полях медианой
    strategy = [['car_make', 'car_model', 'car_gen', 'eng_type'],
                ['car_make', 'car_model', 'car_gen'],
                ['car_make', 'car_model', 'eng_type'],
                ['car_make', 'car_model'],
                ['car_type', 'class_auto', 'eng_type'],
                ['car_type', 'class_auto'],
                ['car_type', 'eng_type'],
                ['car_type'],
                ['class_auto', 'eng_type'],
                ['class_auto']
                ]

    target_columns = ['clearence',
                      'v_bag',
                      'v_tank',
                      'curb_weight',
                      'max_speed',
                      'acceleration',
                      'fuel_cons',
                      'max_torq']

    for target_column in target_columns:
        for columns in strategy:
            pipelines.append(((f'group_median_imputer_{target_column}_{columns}',
                               GroupMedianImputer(group_cols=columns, target_col=target_column))))

    # Добавляем пайплайн для замены пропусков в категориальных полях модой
    strategy = [['car_make', 'car_model', 'car_gen', 'eng_type'],
                ['car_make', 'car_model', 'car_gen'],
                ['car_make', 'car_model', 'eng_type'],
                ['car_make', 'car_model'],
                ['car_type', 'class_auto', 'eng_type'],
                ['car_type', 'class_auto'],
                ['car_type', 'eng_type'],
                ['car_type'],
                ['class_auto', 'eng_type'],
                ['class_auto']
                ]

    target_columns = ['rear_brakes',
                      'max_speed',
                      'fuel_brand',
                      'engine_loc1',
                      'engine_loc2',
                      'turbocharg']

    for target_column in target_columns:
        for columns in strategy:
            pipelines.append(((f'group_moda_imputer_{target_column}_{columns}',
                               GroupModeImputer(group_cols=columns, target_col=target_column))))

    # Добавляем пайплайн для удаления полей, преобразования car_type и count_owner
    drop_columns = ['car_model', 'car_gen', 'car_compl']

    pipelines.append(("column_dropper_transformer", columnDropperTransformer(drop_columns)))
    pipelines.append(("drop_duplicate", DropDuplicate()))
    pipelines.append(("reset_index", resetIndex()))

    # Создаем пайплайн
    pipeline = Pipeline(steps=pipelines)

    # Обучение Pipeline
    transformed_train = pipeline.fit_transform(df_train)

    # Преобразование тестовых данных
    transformed_test = pipeline.transform(df_test)



    return transformed_train, transformed_test