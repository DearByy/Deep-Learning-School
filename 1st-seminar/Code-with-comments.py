import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Загрузка данных
data = pd.read_csv('./AppleStore.csv')

# Определение числовых колонок
num_cols = [
    'size_bytes',
    'price',
    'rating_count_tot',
    'rating_count_ver',
    'sup_devices.num',
    'ipadSc_urls.num',
    'lang.num',
    'cont_rating',
]

# Определение категориальных колонок
cat_cols = [
    'currency',
    'prime_genre'
]

# Целевая колонка
target_col = 'user_rating'

# Выбор нужных колонок из DataFrame
cols = num_cols + cat_cols + [target_col]
data = data[cols]

# Преобразование колонки cont_rating к числовому типу
data['cont_rating'] = data['cont_rating'].str.slice(0, -1).astype(int)

# Удаление колонки currency
data = data.drop(columns=['currency'])
cat_cols.remove('currency')

# Добавление новой бинарной колонки is_free
data['is_free'] = data['price'] == 0
cat_cols.append('is_free')

# Преобразование категориальных колонок в дамми-переменные
data = pd.get_dummies(data, columns=cat_cols)

# Обновление списка категориальных колонок
cat_cols_new = []
for col_name in cat_cols:
    cat_cols_new.extend(filter(lambda x: x.startswith(col_name), data.columns))
cat_cols = cat_cols_new

# Стандартизация данных
scaler = StandardScaler()
scaler.fit(data[num_cols + cat_cols])

# Преобразование данных
X = scaler.transform(data[num_cols + cat_cols])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, data[target_col], test_size=0.2, random_state=102)

# Определение функции для вывода метрик
def print_metrics(y_preds, y):
    print(f'R^2: {r2_score(y, y_preds)}')  # Порядок аргументов в r2_score должен быть y и y_preds
    print(f'MSE: {mean_squared_error(y, y_preds)}')  # Порядок аргументов в mean_squared_error должен быть y и y_preds

# Обучение и оценка линейной регрессии
lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds_lr = lr.predict(X_test)
print("Linear Regression:")
print_metrics(y_preds_lr, y_test)

# Обучение и оценка KNN регрессии
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_preds_knn = knn.predict(X_test)
print("KNN Regression:")
print_metrics(y_preds_knn, y_test)

print()
#Кросс-валидация
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate

cross_validate(LinearRegression(), X, data[target_col], cv=5, #кол-во фолдов, те частей на которые разделяется датасет, перед тем как на всех частях кроме одной обучиться, а на последней посмотреть качество
               scoring={'r2_score': make_scorer(r2_score),
                        'mean_squared_error': make_scorer(mean_squared_error)})

cross_validate(KNeighborsRegressor(), X, data[target_col], cv=5,
               scoring={'r2_score': make_scorer(r2_score, ),
                        'mean_squared_error': make_scorer(mean_squared_error)})

# поиск гиперпараметров GridSearchCV
from sklearn.model_selection import GridSearchCV
gbr_grid_search = GridSearchCV(KNeighborsRegressor(),
                               [{'n_neighbors': [1, 2, 3, 4, 6, 8, 10, 15]}],
                               cv=5,
                               scoring=make_scorer(mean_squared_error),
                               verbose=10)

gbr_grid_search.fit(X_train, y_train)
print(gbr_grid_search.best_params_)  # Это атрибут объекта GridSearchCV, который содержит наилучшую комбинацию гиперпараметров, найденную в процессе поиска.
print(gbr_grid_search.best_score_)   # Это атрибут объекта GridSearchCV, который содержит наилучший результат метрики, достигнутый при использовании наилучшей комбинации гиперпараметров.
print(gbr_grid_search.best_estimator_)  # Это атрибут объекта GridSearchCV, который содержит модель с наилучшими гиперпараметрами, найденными в процессе поиска

best_model = gbr_grid_search.best_estimator_
predictions = best_model.predict(X_test)
print_metrics(y_test,predictions)
