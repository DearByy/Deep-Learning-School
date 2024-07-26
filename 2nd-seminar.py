import numpy as np
import pandas as pd
import scipy.linalg as sla
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge


class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept#если true - то добавляем bias(свободный член)

    def fit(self, X, y):
        # Принимает на вход X, y и вычисляет веса по данной выборке
        # Не забудьте про фиктивный признак равный 1

        n, k = X.shape # n - количество образцов(строчки), k - количество признаков(столбцы)

        X_train = X
        if self.fit_intercept:
            # Добавляем столбец единиц для смещения в конец матрицы
            X_train = np.hstack((X, np.ones((n, 1)))) #np.ones((n, 1) - создание массива заполненного единицами размерностью n*1
            #np.hstack - объединение массивов ( массив из единиц добавляется справа )

         # Вычисляем веса по формуле метода наименьших квадратов
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y

        return self

    def predict(self, X):
        # Принимает на вход X и возвращает ответы модели
        # Не забудьте про фиктивный признак равный 1
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1)))) # Добавляем столбец единиц для смещения, где n - это кол-во строчек

            # Вычисляем предсказанные значения
            y_pred = X_train @ self.w

        return y_pred

    def get_weights(self):
        return self.w

#ГЕНЕРАЦИЯ ИСКУССТВЕННЫХ ДАННЫХ

def linear_expression(x):
    return 5 * x + 6 #наша зависимость

# по признакам сгенерируем значения таргетов с некоторым шумом
objects_num = 50
X = np.linspace(-5, 5, objects_num) #Создали массив признаков X, состоящий из 50 объектов в диапазоне [-5;5]
y = linear_expression(X) + np.random.randn(objects_num) * 5 #Каждый элемент массива X прогоняем через функцию 5 * x + 6 и добавляем какой-то шум в диапазоне от 0 до 5 (из-за умножения)(здесь + складывает каждый элемент первого слагаемого с каждый элементом второго слагаемого)


# выделим половину объектов на тест
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
# обучим нашу реалезацию модели на трейне и предскажем результаты на тесте
regressor = MyLinearRegression() #создаем экзепляр нашего кастом класса

regressor.fit(X_train[:, np.newaxis], y_train)
# Когда мы используем X_train[:, np.newaxis], мы добавляем новую ось, и форма массива изменяется:
#X_train - это обычный массив с 1 строчкой(одномерный). После этой строчки мы решейпим его(делаем двумерным)
# Также можно было написать  X.reshape(-1, 1). Это необходимо, так как многие алгоритмы машинного обучения ожидают входные данные в виде двумерного массива (матрицы) размерностью (n_samples, n_features).
#При этом y_train должен быть одномерным массивом
#Линейная регрессия использует матричное умножение, чтобы найти веса. В этом случае, X_train должен быть двумерным массивом, чтобы умножение было корректным. y_train остается одномерным, потому что мы умножаем матрицу на вектор, что дает нам вектор предсказаний.

w = regressor.get_weights()#Получаем веса, полученные после обучения
predictions = regressor.predict(X_test[:, np.newaxis])#предсказываем результаты

#X - одномерный массив сгенерированных признаков (от -5 до 5) 50 признаков
#X[:, np.newaxis] ЛИБО X.reshape(-1,1)  - поворачиваем массив(теперь 50 строчек и 1 столбец). Аргументы решейпа: -1 - кол-во строчек столжно быть расчитано автоматически. 1 - кол- во столбцов в новом массиве = 1
#np.hstack((X[:, np.newaxis], np.ones((50, 1)))) - соединяем наш массив признаков с массивом из единичек размером 50*1. Новый массив прибаляется справа

#Сравним нашу реализацию с cklearn

sk_reg = LinearRegression().fit(X_train[:, np.newaxis], y_train)#можно было написать sk_reg = LinearRegression(); sk_reg.fit(...)

'''plt.figure(figsize=(10, 7))#создали фигуру размером 10 на 7 дюймов
plt.plot(X, linear_expression(X), label='real', c='g')#Создали зеленую ('g' = green) линию с уравнением linear_expression(X) (то есть 5 * X + 6) и добавили метку real (которая будет показываться слева сверху)

plt.scatter(X_train, y_train, label='train')#Создает точки для тренировочных данных X_train и y_train и добавляет метку
plt.scatter(X_test, y_test, label='test')#Создает разбросанные точки для тестовых данных X_test и y_test.
plt.plot(X, regressor.predict(X[:, np.newaxis]), label='ours', c='r')# строит прямую предсказанную нашей моделью
plt.plot(X, sk_reg.predict(X[:, np.newaxis]), label='sklearn', c='cyan', linestyle=':')#строит прямую предсказанную sklearn моделью

plt.title("Different Prediction")
plt.ylabel('target')
plt.xlabel('feature')
plt.grid(alpha=0.2)
plt.legend()
plt.show()'''

#Результаты
from sklearn.metrics import mean_squared_error

train_predictions = regressor.predict(X_train[:, np.newaxis])
test_predictions = regressor.predict(X_test[:, np.newaxis])

'''print('Результаты обычных(матричных) регрессоров на ТРЕНИРОВОЧНОЙ и ТЕСТОВОЙ выборках')
print('Train MSE: ', mean_squared_error(y_train, train_predictions))
print('Test MSE: ', mean_squared_error(y_test, test_predictions),'\n')'''


#РЕАЛИЗУЕМ ГРАДИЕНТНУЮ ОПТИМИЗАЦИЮ
class MyGradientLinearRegression(MyLinearRegression):#класс наследник MyLinearRegression
    def __init__(self, **kwargs):#Определяет конструктор для класса MyGradientLinearRegression, который принимает произвольное количество именованных аргументов.
        super().__init__(**kwargs) # передает именные параметры родительскому конструктору
        self.w = None

    def fit(self, X, y, lr=0.01, max_iter=100): #lr: скорость обучения (по умолчанию 0.01), max_iter: максимальное количество итераций (по умолчанию 100).
        # Принимает на вход X, y и вычисляет веса по данной выборке
        # Не забудьте про фиктивный признак равный 1!

        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1 if self.fit_intercept else k)#если веса не инициализированны, они инициализируются случайными значениями
            #Если fit_intercept равно True, добавляется дополнительный вес для смещения (bias).

        X_train = np.hstack((X, np.ones((n, 1)))) if self.fit_intercept else X #Добавляем столбец единиц к X, если fit_intercept равно True

        self.losses = []# Инициализирует пустой список для хранения значений функции потерь на каждой итерации. Сделано чтобы мы могли сравнивать эту функцию на различных иттерациях
        #Если функция потерь перестает значительно уменьшаться, это может указывать на то, что модель достигла локального минимума или оптимального решения.

        for iter_num in range(max_iter):
            y_pred = self.predict(X)#Вычисляет предсказанные значения модели для текущих весов.
            self.losses.append(mean_squared_error(y_pred, y))#Вычисляет среднеквадратичную ошибку (MSE) между предсказанными и реальными значениями и добавляет ее в список losses

            grad = self._calc_gradient(X_train, y, y_pred)#Вычисляет градиент функции потерь по весам с использованием метода _calc_gradient

            assert grad.shape == self.w.shape, f"gradient shape {grad.shape} is not equal weight shape {self.w.shape}"#Проверяет, что форма градиента совпадает с формой весов. Если нет, выбрасывает ошибку
            self.w -= lr * grad#Обновляет веса, вычитая градиент, умноженный на скорость обучения.

        return self

    def _calc_gradient(self, X, y, y_pred):#Метод для вычисления градиента
        grad = 2 * (y_pred - y)[:, np.newaxis] * X#(y_pred - y) — это вектор ошибок, [np.newaxis] добавляет новую ось, чтобы можно было умножить на матрицу X(ведь y_pred - y - это одномерный массив), а X — это двумерный массив признаков с формой (n, k).
        # Когда вы добавляете новую ось с помощью [:, np.newaxis], форма массива ошибок изменяется с (n,) на (n, 1)(те он тсновится двумерным). Теперь массив ошибок можно элементно умножить на матрицу признаков X(причем это не матричное умножение, а элементарное(умножение соотвествующих элементов)

        grad = grad.mean(axis=0)#Усредняет градиенты по всем образцам, чтобы получить окончательный градиент
        return grad

    def get_losses(self):#метод получения списка значений потерь
        return self.losses

#ТЕСТ

gradregressor = MyGradientLinearRegression(fit_intercept=True)

gradregressor.fit(X_train[:, np.newaxis], y_train, max_iter=10000)

l = gradregressor.get_losses()

predictions = gradregressor.predict(X[:, np.newaxis])

'''plt.figure(figsize=(10, 5))
plt.plot(X, linear_expression(X), label='real', c='g')

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, predictions, label='Gradient predicted 10000 steps', c='r')
plt.plot(X, sk_reg.predict(X[:, np.newaxis]), label='sklearn predicted', c='cyan', linestyle=':')#строит прямую предсказанную sklearn моделью


plt.grid(alpha=0.2)
plt.legend()
plt.show()'''

#График Лосса
'''plt.figure(figsize=(10, 7))

plt.plot(l)

plt.title('Gradient descent learning')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.ylim(bottom=0)
plt.grid(alpha=0.2)
plt.show()'''

# Второй градРегрессор с меньшим кол-вом иттераций
gradregressor2 = MyGradientLinearRegression()

gradregressor2.fit(X_train[:, np.newaxis], y_train, max_iter=100)

predictions2 = gradregressor2.predict(X[:, np.newaxis])
'''print('разница между ГрадРегрессорами на ТРЕНИРОВОЧНОЙ и ТЕСТОВОЙ выборках','\n')
print('gradregressor train MSE: ', mean_squared_error(y_train, gradregressor.predict(X_train[:, np.newaxis])))
print('gradregressor2 train MSE: ', mean_squared_error(y_train, gradregressor2.predict(X_train[:, np.newaxis])))

print('gradregressor test MSE: ', mean_squared_error(y_test, gradregressor.predict(X_test[:, np.newaxis])))
print('gradregressor2 test MSE: ', mean_squared_error(y_test, gradregressor2.predict(X_test[:, np.newaxis])),'\n')'''




'''plt.figure(figsize=(15, 10))
plt.plot(X, linear_expression(X), label='real', c='g')
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, predictions, label='Gradient predicted 10000 steps', c='r')
plt.plot(X, predictions2, label='Gradient2 predicted 100 steps', c='cyan', linestyle=':')
plt.title("Разные град регрессоры")
plt.ylabel('target')
plt.xlabel('feature')
plt.grid(alpha=0.2)
plt.legend()
plt.show()'''


# Стохастический градиентный спуск

class MySGDLinearRegression(MyGradientLinearRegression):
    def __init__(self, n_sample=10, **kwargs):
        super().__init__(**kwargs) # передает именные параметры родительскому конструктору
        self.w = None
        self.n_sample = n_sample

    def _calc_gradient(self, X, y, y_pred):
        # Главное отличие в SGD -- это использование подвыборки для шага оптимизации
        inds = np.random.choice(np.arange(X.shape[0]), size=self.n_sample, replace=False) # X.shape[0] возвращает кол-во строк (образцов) в массиве X
        # replace=False говорит о том, что каждый элемент может быть выбран только один раз,те все элементы уникальны
        # При чем при каждом вызове функции _calc_gradient она выбирает новые экземпляры. Это позволяет избежать локальных минимумов

        grad = 2 * (y_pred[inds] - y[inds])[:, np.newaxis] * X[inds]
        grad = grad.mean(axis=0)

        return grad

# ТЕСТИРОВАНИЕ

SGD_regressor = MySGDLinearRegression(fit_intercept=True)

SGD_l = SGD_regressor.fit(X_train[:, np.newaxis], y_train, max_iter=10000).get_losses()

SGD_predictions = SGD_regressor.predict(X_test[:, np.newaxis])

'''
plt.figure(figsize=(10, 7))
plt.plot(X, linear_expression(X), label='real', c='g')

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.plot(X, SGD_regressor.predict(X[:, np.newaxis]), label='predicted by SGD', c='cyan', linestyle=':')
plt.plot(X, predictions, label='Gradient predicted 10000 steps', c='r')


plt.grid(alpha=0.2)
plt.legend()
plt.show()'''

'''print('Разница между SGD и обычной регрессией:','\n')
print('gradregressor train MSE: ', mean_squared_error(y_train, gradregressor.predict(X_train[:, np.newaxis])))
print('SGD_gradregressor train MSE: ', mean_squared_error(y_train, SGD_regressor.predict(X_train[:, np.newaxis])))

print('gradregressor test MSE: ', mean_squared_error(y_test, gradregressor.predict(X_test[:, np.newaxis])))
print('SGD_gradregressor test MSE: ', mean_squared_error(y_test, SGD_regressor.predict(X_test[:, np.newaxis])),'\n')'''

# График ошибки
'''plt.figure(figsize=(10, 7))
plt.plot(SGD_l)
plt.title('Gradient descent learning')
plt.ylabel('loss')
plt.xlabel('iteration')
plt.grid(alpha=0.2)
plt.show()
'''
# Протестируем меняя количество элементов для расчета градиента
n_samples = [1, 2, 4,6,8]

plt.figure(figsize=(10, 7))

for ns in n_samples:
    # Здесь по сути используется метод .fit родительского класса. В дочернем классе мы перегрузили только метод _calc_gradient
    l = MySGDLinearRegression(fit_intercept=True, n_sample=ns).fit(
        X_train[:, np.newaxis],
        y_train,
        lr=5e-3,
        max_iter=150,
    ).get_losses()
    plt.plot(l, alpha=0.5, label=f'{ns} mini-batch size')

plt.title('Gradient descent learning')
plt.ylabel('loss')
plt.xlabel('iteration')

plt.legend()
plt.ylim((0, 150))
plt.grid(alpha=0.2)
plt.show()

#Реализация логистической регрессии

def logit(x, w): # обычная логическая ргерессия: Умножаем вектор весов на вектор признаков
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, max_iter=100, lr=0.1):
        # Принимает на вход X, y и вычисляет веса по данной выборке.
        # Множество допустимых классов: {1, -1}
        # Не забудьте про фиктивный признак равный 1!

        n, k = X.shape

        if self.w is None:
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []

        for iter_num in range(max_iter):
            z = sigmoid(logit(X_train, self.w))
            grad = np.dot(X_train.T, (z - y)) / len(y)

            self.w -= grad * lr

            losses.append(self.__loss(y, z))

        return losses

    def predict_proba(self, X): #Предсказание вероятности того, что объект принадлежит классу 1
        # Принимает на вход X и возвращает ответы модели
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def predict(self, X, threshold=0.5): # True - метод возвращает то, к каком классу с большей вероятность приналежит объект
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w

    def __loss(self, y, p): # В функции Лосса есть логарифм. Если мы засунем в логарифм в питоне 0, то он выдаст NONE. Чтобы этого избежать мы заменяем 0 на маленькое положиельное, а 1 на близкое к 1 число
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# Создание выборки из двух классов
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=[[-2, 0.5], [2, -0.5]], cluster_std=1, random_state=42)

colors = ("red", "green")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([0, 1]):
    colored_y[y == cl] = str(colors[i])

plt.figure(figsize=(15, 10))
plt.scatter(X[:, 0], X[:, 1], c=colored_y)
plt.show()

#Обучение логистической регрессии
clf = MyLogisticRegression()

clf.fit(X, y, max_iter=1000)

#Строим график
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.figure(figsize=(15, 8))

eps = 0.1
xx, yy = np.meshgrid(np.linspace(np.min(X[:, 0]) - eps, np.max(X[:, 0]) + eps, 500),
                     np.linspace(np.min(X[:, 1]) - eps, np.max(X[:, 1]) + eps, 500))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=colored_y)

# И тепловую карту
colors = ("magenta", "green")
colored_y = np.zeros(y.size, dtype=str)

for i, cl in enumerate([0, 1]):
    colored_y[y == cl] = str(colors[i])

plt.figure(figsize=(15, 8))

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.get_cmap('viridis'))

plt.scatter(X[:, 0], X[:, 1], c=colored_y)
plt.colorbar()
plt.show()
