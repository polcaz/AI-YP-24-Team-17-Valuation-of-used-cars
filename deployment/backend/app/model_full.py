import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE, mean_absolute_error as MAE
import random

def full_model(train, test, alpha):

    # Преобразуем и разделяем признаки и таргет
    X_train = train.copy()
    y_train = np.log(train['car_price'])
    X_test = test.copy()
    y_test = np.log(test['car_price'])
    X_train = X_train.drop(['car_price'], axis=1)
    X_test = X_test.drop(['car_price'], axis=1)

    # Собственно, наш пайплайн
    cat_features_mask = (X_train.dtypes == "object").values

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', min_frequency=4, max_categories=140, handle_unknown='ignore'))
        # OHE-кодирование
    ])

    numerical_transformer = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2)),  # Poly-кодирование
        ('scaler', StandardScaler())  # Стандартизируем
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, X_train.columns[cat_features_mask]),
            # Обрабатываем категориальные признаки
            ('num', numerical_transformer, X_train.columns[~cat_features_mask])  # Обрабатываем числовые признаки
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha))  # Используем Ridge
    ])

    model = pipeline.fit(X_train, y_train)  # Обучаем модель

    return model