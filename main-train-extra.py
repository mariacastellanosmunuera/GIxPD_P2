import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from joblib import dump

MODEL_PATH = os.environ["MODEL_PATH"]

np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Transforma las características en un conjunto de características polinómicas
degree = 4
poly = PolynomialFeatures(degree=degree)
train_x_poly = poly.fit_transform(train_x.reshape(-1, 1))

# Ajusta un modelo de regresión lineal
model = LinearRegression()
model.fit(train_x_poly, train_y)

# Calcula el coeficiente de determinación (R-squared) en el conjunto de prueba
test_x_poly = poly.transform(test_x.reshape(-1, 1))
y_pred = model.predict(test_x_poly)
r2 = r2_score(test_y, y_pred)
print("Model trained successfully")
print("Model Score:", r2)

# Guarda el modelo entrenado
dump(model,MODEL_PATH)

