import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

def preproc(df):

    # Разделяем ИСХОДНЫЕ ДАННЫЕ на тренировочную и тестовую выборки
    df_train, df_test = train_test_split(df, train_size=0.75)

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

    # Готовим данные для обработки
    drop_columns = ['Unnamed: 0', 'url_car', 'ann_id', 'ann_date', 'ann_city',
                        'avail', 'original_pts', 'customs', 'link_cpl', 'eng_power_kw',
                        'pow_resrv', 'options', 'condition', 'url_compl', 'gross_weight']

    column_dropper= Pipeline([
        ("column_dropper", columnDropperTransformer(drop_columns))
    ])

    car_type_transformer = Pipeline([
        ("car_type_transformer", columnCarTypeTransformer('car_type'))
    ])

    df_train = column_dropper.fit_transform(df_train)
    car_type_transformer.fit_transform(df_train)
    df_test = column_dropper.fit_transform(df_test)
    car_type_transformer.fit_transform(df_test)


    # Класс для замены полей автомобиля определенным значением по фильтру
    class filterTransformer():

        def __init__(self, filters, column_fix, column_replace, value_replace):
            self.filters = filters
            self.column_fix = column_fix
            self.column_replace = column_replace
            self.value_replace = value_replace

        def transform(self, X, y=None):
            def replace_con(df, filter, column_fix, column_replace, value_replace):
                '''
                Функция получает на вход датасет и по укзанному
                значению фильтра в фильтруемом поле, заменяет пропуски
                в заменяем поле на значение value_replace
                '''
                return df.apply(
                    lambda row: value_replace if pd.isnull(row[column_replace]) & (row[column_fix] == filter) else row[
                        column_replace], axis=1)

            for filter in self.filters:
                X[self.column_replace] = replace_con(X, filter, self.column_fix, self.column_replace, self.value_replace)
            return X

        def fit(self, X, y=None):
            return self

    # Создаем датасеты для замены значений медианами и модами
    def df_median_mode(df, columns):
        '''
        Функция получает на вход датасет и формирует по нему датасет
        с медианами и модами расчитанными в группировке
        по полям в списке columns
        '''
        columns_isnumber = df.select_dtypes([int, float]).columns.to_list()
        df_med_mod = df.copy()

        for column in columns_isnumber:
            df_med_mod[column] = df.groupby(columns)[column].transform('median')

        columns_isobject = df.select_dtypes([object]).columns.to_list()

        for column in columns:
            columns_isobject.remove(column)

        for column in columns_isobject:
            df_med_mod[column] = df.groupby(columns)[column].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)

        df_med_mod = df_med_mod.drop_duplicates(keep='first')

        return df_med_mod

    # медианы и моды в группировке по марке, модели, поколению и типу двигателя
    columns = ['car_make', 'car_model', 'car_gen', 'eng_type']
    df_mmge = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по марке, модели и поколению
    columns = ['car_make', 'car_model', 'car_gen']
    df_mmg = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по марке, модели и типу двигателя
    columns = ['car_make', 'car_model', 'eng_type']
    df_mme = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по марке и модели
    columns = ['car_make', 'car_model']
    df_mm = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по типу кузова, классу автомобиля и типу двигателя
    columns = ['car_type', 'class_auto', 'eng_type']
    df_tce = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по типу кузова и классу автомобиля
    columns = ['car_type', 'class_auto']
    df_tc = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по типу кузова
    columns = ['car_type']
    df_t = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по классу автомобиля
    columns = ['class_auto']
    df_c = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по типу кузова и типу двигателя
    columns = ['car_type', 'eng_type']
    df_te = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # медианы и моды в группировке по классу автомобиля и типу двигателя
    columns = ['class_auto', 'eng_type']
    df_ce = [df_median_mode(df_train, columns).sort_values(by=columns), columns]

    # Класс для замещения пропуска в поле медианой или модой
    class medModTransformer():
        def __init__(self, df_med_mods, column_replace):
            self.df_med_mods = df_med_mods
            self.column_replace = column_replace
        def transform(self, X, y=None):
            def return_med_mod(df_med_mod, row, column_replace):
                '''
                Функция принимает на вход датасет
                с медиананами, модами и полями группировки (до 5 полей),
                значения записи для фильтрации, поле для замены значения
                и возвращает значение поля для указанных параметров
                '''
                df = df_med_mod[0]
                column_group = df_med_mod[1]

                if len(column_group) == 5:
                    try:
                        return df[(df[column_group[0]] == row[column_group[0]]) &
                                (df[column_group[1]] == row[column_group[1]]) &
                                (df[column_group[2]] == row[column_group[2]]) &
                                (df[column_group[3]] == row[column_group[3]]) &
                                (df[column_group[4]] == row[column_group[4]])][column_replace].values[0]
                    except:
                        return np.nan
                elif len(column_group) == 4:
                    try:
                        return df[(df[column_group[0]] == row[column_group[0]]) &
                                (df[column_group[1]] == row[column_group[1]]) &
                                (df[column_group[2]] == row[column_group[2]]) &
                                (df[column_group[3]] == row[column_group[3]])][column_replace].values[0]
                    except:
                        return np.nan
                elif len(column_group) == 3:
                    try:
                        return df[(df[column_group[0]] == row[column_group[0]]) &
                                (df[column_group[1]] == row[column_group[1]]) &
                                (df[column_group[2]] == row[column_group[2]])][column_replace].values[0]
                    except:
                        return np.nan
                elif len(column_group) == 2:
                    try:
                        return df[(df[column_group[0]] == row[column_group[0]]) &
                                (df[column_group[1]] == row[column_group[1]])][column_replace].values[0]
                    except:
                        return np.nan
                elif len(column_group) == 1:
                    try:
                        return df[(df[column_group[0]] == row[column_group[0]])][column_replace].values[0]
                    except:
                        return np.nan

            for df_med_mod in self.df_med_mods:
                X[self.column_replace] = X.apply(lambda row: return_med_mod(df_med_mod, row, self.column_replace) if pd.isnull(row[self.column_replace]) else row[self.column_replace], axis=1)
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

    # Класс для удаления дубликатов
    class dropDuplicate():

        def transform(self, X, y=None):
            column_features = df_train.columns[df_train.any()].to_list()
            columns_remove = ['car_price', 'car_model', 'car_gen', 'car_compl']
            for column_remove in columns_remove:
                column_features.remove(column_remove)
            X = X.drop_duplicates(subset=column_features, keep='first')
            X = X.reset_index(drop=True)
            return X

        def fit(self, X, y=None):
            return self

    # Класс для обновления индекса
    class resetIndex():

        def transform(self, X, y=None):
            X = X.reset_index(drop=True)
            return X

        def fit(self, X, y=None):
            return self

    # Создаем преобразователи данных
    column_dropper = Pipeline([
        ("column_dropper", columnDropperTransformer(['Unnamed: 0', 'url_car', 'ann_id', 'ann_date', 'ann_city',
                                                     'avail', 'original_pts', 'customs', 'link_cpl', 'eng_power_kw',
                                                     'pow_resrv', 'options', 'condition', 'url_compl', 'gross_weight']))
    ])

    car_type_transformer = Pipeline([
        ("car_type_transformer", columnCarTypeTransformer('car_type'))
    ])

    class_auto_transformer = Pipeline(steps=[
        ("transform_class_auto_mmge", medModTransformer([df_mmge, df_mmg, df_mme, df_mm], 'class_auto')),
        ("transform_class_auto_to_M", filterTransformer(['Фургон'], 'car_type', 'class_auto', 'M')),
        ("transform_class_auto_tce", medModTransformer([df_tce, df_tc, df_te, df_t, df_ce, df_c], 'class_auto')),
        ("transform_class_auto_to_F", filterTransformer(['Лимузин'], 'car_type', 'class_auto', 'F'))
    ])

    eng_size_transformer= Pipeline([
        ("eng_size_transformer", filterTransformer(['Электро'], 'eng_type', 'eng_size', 0))
    ])

    clearence_transformer= Pipeline([
        ("clearence_transformer", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'clearence'))
    ])

    v_bag_transformer = Pipeline(steps=[
        ("transform_v_bag_class_auto_to_0", filterTransformer(['S'], 'class_auto', 'v_bag', 0)),
        ("transform_v_bag_car_type_to_0", filterTransformer(['Кабриолет', 'Купе', 'Пикап Двойная кабина', 'Пикап Одинарная кабина',
                                                             'Пикап Полуторная кабина', 'Родстер', 'Тарга'],
                                                             'car_type', 'v_bag', 0)),
        ("transform_v_bag_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'v_bag'))
    ])

    v_tank_transformer = Pipeline(steps=[
        ("transform_v_tank_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'v_tank', 0)),
        ("transform_v_tank_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'v_tank'))
    ])

    curb_weight_transformer = Pipeline([
        ("curb_weight_transformer", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'curb_weight'))
    ])

    rear_brakes_transformer = Pipeline([
        ("rear_brakes_transformer", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'rear_brakes'))
    ])

    max_speed_transformer= Pipeline([
        ("max_speed_transformer", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'max_speed'))
    ])

    acceleration_transformer= Pipeline([
        ("acceleration_transformer", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'acceleration'))
    ])

    fuel_cons_transformer = Pipeline(steps=[
        ("transform_fuel_cons_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'fuel_cons', 0)),
        ("transform_fuel_cons_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'fuel_cons'))
    ])

    fuel_brand_transformer = Pipeline(steps=[
        ("transform_fuel_brand_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'fuel_brand', 'Nan')),
        ("transform_fuel_brand_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'fuel_brand'))
    ])

    cyl_count_transformer = Pipeline(steps=[
        ("transform_cyl_count_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'cyl_count', 0)),
        ("transform_cyl_count_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'cyl_count'))
    ])

    engine_loc1_transformer = Pipeline(steps=[
        ("transform_engine_loc1_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'engine_loc1', 'Nan')),
        ("transform_engine_loc1_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'engine_loc1'))
    ])

    engine_loc2_transformer = Pipeline(steps=[
        ("transform_engine_loc2_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'engine_loc2', 'Nan')),
        ("transform_engine_loc2_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'engine_loc2'))
    ])

    turbocharg_transformer = Pipeline(steps=[
        ("transform_turbocharg_eng_type_to_0", filterTransformer(['Электро'], 'eng_type', 'turbocharg', 'Nan')),
        ("transform_turbocharg_mmg", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'turbocharg'))
    ])

    max_torq_transformer= Pipeline([
        ("transformer_max_torq", medModTransformer([df_mmge, df_mmg, df_mme, df_mm, df_tce, df_tc, df_te, df_t, df_ce, df_c], 'max_torq'))
    ])

    count_owner_transformer = Pipeline([
        ("count_owner_transformer", classOwnsTransformer())
    ])

    drop_duplicate = Pipeline([
        ("drop_duplicate", dropDuplicate())
    ])

    reset_index = Pipeline([
        ("reset_index", resetIndex())
    ])

    # Объединяем преобразователи для трэйна
    column_null_transformer = Pipeline(steps=[
        ('class_auto_transformer', class_auto_transformer),
        ('eng_size_transformer', eng_size_transformer),
        ('clearence_transformer', clearence_transformer),
        ('v_bag_transformer', v_bag_transformer),
        ('v_tank_transformer', v_tank_transformer),
        ('curb_weight_transformer', curb_weight_transformer),
        ('rear_brakes_transformer', rear_brakes_transformer),
        ('max_speed_transformer', max_speed_transformer),
        ('acceleration_transformer', acceleration_transformer),
        ('fuel_cons_transformer', fuel_cons_transformer),
        ('fuel_brand_transformer', fuel_brand_transformer),
        ('cyl_count_transformer', cyl_count_transformer),
        ('engine_loc1_transformer', engine_loc1_transformer),
        ('engine_loc2_transformer', engine_loc2_transformer),
        ('turbocharg_transformer', turbocharg_transformer),
        ('max_torq_transformer', max_torq_transformer),
        ("column_dropper", columnDropperTransformer(['car_model', 'car_gen', 'car_compl'])),
        ("count_owner_transformer", count_owner_transformer),
        ("drop_duplicate", drop_duplicate)
    ])

    # Объединяем преобразователи для теста
    column_null_transformer_test = Pipeline(steps=[
        ('class_auto_transformer', class_auto_transformer),
        ('eng_size_transformer', eng_size_transformer),
        ('clearence_transformer', clearence_transformer),
        ('v_bag_transformer', v_bag_transformer),
        ('v_tank_transformer', v_tank_transformer),
        ('curb_weight_transformer', curb_weight_transformer),
        ('rear_brakes_transformer', rear_brakes_transformer),
        ('max_speed_transformer', max_speed_transformer),
        ('acceleration_transformer', acceleration_transformer),
        ('fuel_cons_transformer', fuel_cons_transformer),
        ('fuel_brand_transformer', fuel_brand_transformer),
        ('cyl_count_transformer', cyl_count_transformer),
        ('engine_loc1_transformer', engine_loc1_transformer),
        ('engine_loc2_transformer', engine_loc2_transformer),
        ('turbocharg_transformer', turbocharg_transformer),
        ('max_torq_transformer', max_torq_transformer),
        ("column_dropper", columnDropperTransformer(['car_model', 'car_gen', 'car_compl'])),
        ("count_owner_transformer", count_owner_transformer),
        ("reset_index", reset_index)
    ])

    # Преобразуем и разделяем признаки и таргет
    train = column_null_transformer.fit_transform(df_train)
    test = column_null_transformer_test.fit_transform(df_test)

    return train, test