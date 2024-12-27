import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FullModel:
    def __init__(self, alpha):
        """
        Инициализация класса модели.

        :param alpha: Коэффициент регуляризации для Ridge-регрессии.
        """
        self.alpha = alpha
        self.pipeline = None
        self.cat_features_mask = None

    def prepare_data(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Подготовка данных: разделение на признаки и целевую переменную.

        :param train: Обучающий датасет.
        :param test: Тестовый датасет.
        :return: Подготовленные X_train, y_train, X_test, y_test.
        """
        X_train = train.copy()
        y_train = np.log(train['car_price'])
        X_test = test.copy()
        y_test = np.log(test['car_price'])

        X_train = X_train.drop(['car_price'], axis=1)
        X_test = X_test.drop(['car_price'], axis=1)

        self.cat_features_mask = (X_train.dtypes == "object").values
        return X_train, y_train, X_test, y_test

    def build_pipeline(self, X_train: pd.DataFrame):
        """
        Создание пайплайна для обработки данных и обучения модели.

        :param X_train: Обучающий датасет (признаки).
        """
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', min_frequency=4, max_categories=140, handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, X_train.columns[self.cat_features_mask]),
                ('num', numerical_transformer, X_train.columns[~self.cat_features_mask])
            ]
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=self.alpha))
        ])

    def build_pipeline_cat(self, X_train: pd.DataFrame):
        """
        Создание пайплайна для обработки данных и обучения модели.

        :param X_train: Обучающий датасет (признаки).
        """
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', min_frequency=4, max_categories=140, handle_unknown='ignore'))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, X_train.columns[self.cat_features_mask]),
                ('num', numerical_transformer, X_train.columns[~self.cat_features_mask])
            ]
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=self.alpha))
        ])

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучение модели.

        :param X_train: Признаки обучающего набора.
        :param y_train: Целевая переменная обучающего набора.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call `build_pipeline` first.")
        self.pipeline.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание целевой переменной.

        :param X: Датасет для предсказания.
        :return: Предсказания модели.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call `build_pipeline` first.")
        return self.pipeline.predict(X)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Оценка модели на тестовом наборе данных.

        :param X_test: Признаки тестового набора.
        :param y_test: Целевая переменная тестового набора.
        :return: Среднеквадратичная ошибка.
        """
        predictions = self.predict(X_test)
        # return mean_squared_error(y_test, predictions)
        return r2_score(y_test, predictions)

    def learning_curve(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                       steps: int = 10):
        """
        Построение кривой обучения.

        :param X_train: Признаки обучающего набора.
        :param y_train: Целевая переменная обучающего набора.
        :param X_test: Признаки тестового набора.
        :param y_test: Целевая переменная тестового набора.
        :param steps: Количество шагов для построения кривой.
        :return: Списки ошибок для обучающего и тестового наборов.
        """
        train_errors = []
        test_errors = []
        data_sizes = np.linspace(0.1, 1.0, steps)

        for size in data_sizes:
            X_partial, _, y_partial, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)
            self.train_model(X_partial, y_partial)

            train_predictions = self.predict(X_partial)
            test_predictions = self.predict(X_test)

            train_errors.append(mean_squared_error(y_partial, train_predictions, squared=False))
            test_errors.append(mean_squared_error(y_test, test_predictions, squared=False))

        return data_sizes, train_errors, test_errors