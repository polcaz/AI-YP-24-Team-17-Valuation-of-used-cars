from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Класс для удаления столбцов из датасета
class columnDropperTransformer():
    def __init__(self,columns):
        self.columns=columns
    def transform(self,X,y=None):
        X = X.drop(self.columns,axis=1)
        return X
    def fit(self, X, y=None):
        return self     # Нет необходимости обучать
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
        return self     # Нет необходимости обучать
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, year_column: str = "year", new_column: str = "car_age"):
        """
        Инициализация трансформера.

        :param year_column: Год выпуска автомобиля.
        :param new_column: Возраст автомобиля.
        """
        self.year_column = year_column
        self.new_column = new_column

    def fit(self, X, y=None):
        return self     # Нет необходимости обучать

    def transform(self, X):
        """
        Извлекает числа из начала строки для указанных столбцов.

        :param X: Входной DataFrame с годом производства.
        :return: Преобразованный DataFrame c возрастом автомобиля.
        """
        X = X.copy()
        current_year = pd.Timestamp.now().year
        X[self.new_column] = current_year - X[self.year_column]
        return X.drop(columns=[self.year_column])
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, delta):
        """
        Инициализация трансформера.

        :param year_column: Год выпуска автомобиля.
        :param new_column: Возраст автомобиля.
        """
        self.column = column
        self.delta = delta

    def fit(self, X, y=None):
        return self     # Нет необходимости обучать

    def transform(self, X):
        """
        Извлекает числа из начала строки для указанных столбцов.

        :param X: Входной DataFrame с годом производства.
        :return: Преобразованный DataFrame c возрастом автомобиля.
        """
        X = X.copy()
        X[self.column] = np.log(X[self.column]+self.delta)
        return X
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MultiplyDivideTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col1: str, col2: str, operation: str = "mul"):
        """
        Инициализация трансформера.

        :param col1: Название первого столбца.
        :param col2: Название второго столбца.
        :param operation: Операция ('multiply' или 'divide').
        :param new_column: Название нового столбца.
        """
        self.col1 = col1
        self.col2 = col2
        self.operation = operation
        self.new_column = f'{col1}_{operation}_{col2}'

    def fit(self, X, y=None):
        return self  # Нет необходимости в обучении

    def transform(self, X):
        """
        Выполняет операцию умножения или деления и создаёт новый столбец.

        :param X: Входной DataFrame.
        :return: Преобразованный DataFrame с новым столбцом.
        """
        X = X.copy()
        if self.operation == "mul":
            X[self.new_column] = X[self.col1] * X[self.col2]
        elif self.operation == "div":
            X[self.new_column] = X[self.col1] / (X[self.col2] + 1)
        else:
            raise ValueError("Операция должна быть 'mul' или 'div'")
        return X

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
                .astype(str)             # Преобразование в строку
                .str.extract(r'^(\d+)')  # Извлечение первого числа в начале строки
                .astype(float)           # Преобразование к числовому типу
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

    def transform(self,X,y=None):
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