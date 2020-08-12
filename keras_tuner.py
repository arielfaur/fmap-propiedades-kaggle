# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
# %matplotlib inline

from urllib.parse import urljoin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

import tensorflow as tf
import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
import tensorflow_docs.modeling

# from kerastuner.tuners import RandomSearch, BayesianOptimization
# from kerastuner import Objective
import kerastuner

import pyreadr
import os
import sys

IS_COLAB = 'google.colab' in sys.modules

# palabras a buscar en columna descripción
TEXT_MINING_KEYWORDS = ['balcon', 'cochera', 'sum', 'parrilla', 'pileta']

# url repositorio
GITHUB_REPOSITORY_URL = 'https://github.com/arielfaur/prediccion-precio-propiedades-fmap-2020/raw/master/'
BASE_URL_DATA = GITHUB_REPOSITORY_URL if IS_COLAB else ''

# global seeds
np.random.seed(1234)
tf.random.set_seed(1234)


def vectorize_descripcion(df: pd.DataFrame):
    # entrenamos el modelo para buscar las keywords
    vectorizer = CountVectorizer(binary=True, strip_accents='unicode')  
    vectorizer.fit(TEXT_MINING_KEYWORDS)

    # hot encoding de los resultados
    desc = vectorizer.transform(df['descripcion'])

    # crear dataframe con resultados
    df_desc = pd.DataFrame(data=desc.toarray(), columns=vectorizer.get_feature_names())
    return pd.concat([df.reset_index(drop=True), df_desc.reset_index(drop=True)], axis=1, sort=False)

def get_outliers(df: pd.DataFrame):
    return df[df_train['id'].isin(['7798', '31417', '10764', '12865', '22126', '49492'])]

def remove_outliers(df: pd.DataFrame):
    outliers = get_outliers(df)
    df.drop(outliers.index, inplace=True)

def remove_duplicates(df: pd.DataFrame):
    df.drop(df[df.duplicated(subset=['descripcion', 'precio'], keep=False)].index, inplace=True)

def preprocessing(df: pd.DataFrame, process_outliers = True):
    # nombres features en minuscula
    df.columns = map(str.lower, df.columns)

    df['barrio'] = df['barrio'].astype('category')
    df['cluster'] = df['cluster'].astype('category')
    df['a_estrenar'] = df['a_estrenar'].astype('category')
    df['lujo'] = df['lujo'].astype('category')
    df['id'] = df['id'].astype('int')
    
    # guardar los id
    ids = df['id']
    
    # eliminar outliers
    if process_outliers:
        remove_outliers(df)

    remove_duplicates(df)
    
    # text mining buscar keywords en descripcion
    df['descripcion'].fillna('', inplace=True)
    df = vectorize_descripcion(df)
    
    # eliminar columnas innecesarias del modelo
    df.drop([
        'banios',
        # 'barrio', 
        'subbarrio',
        'descripcion',
        'dormitorios',
        'p_mt2', 
        'n_bancos',
        'n_barrios_p', 
        'n_bici', 
        'n_boliches', 
        'n_clinicas', 
        'n_comisarias',
        'n_embajadas', 
        'n_gasolina', 
        'n_gastronomica', 
        'n_homicidios',
        'n_hospitales', 
        'n_hurtos', 
        'n_hurtos_auto',
        'n_robo', 
        'n_robo_vehi',
        'n_subte_bocas', 
        'n_uni',
        'cluster',
        # 'terraza',
        'lujo',
        'sup_cubierta'
    ],axis=1, inplace=True, errors='raise')


    return ids, df.set_index('id')


def split_data(df: pd.DataFrame, test_size = 0.05):
    y = df[['precio']]
    X = df.drop(['precio'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=5)
    return X_train, X_test, y_train, y_test


## Preprocesamiento de los features: imputación, estandarización, one-hot encoding
def make_transformer_pipeline():
    numeric_transformer = make_pipeline( 
        SimpleImputer(missing_values = np.nan, strategy='mean'),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        OneHotEncoder(drop='first')
    )

    return make_column_transformer(
        (numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        (categorical_transformer, make_column_selector(dtype_include=['category']))
    )    


# Graficar predicciones
def plot_test_predictions(y_test, y_pred):
    # graficar predicciones contra datos actuales
    fig, ax = plt.subplots(figsize = (18,10))
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Precio test')
    ax.set_ylabel('Precio predicción')
    plt.show()

def read_rds(filename):
    result = pyreadr.read_r(filename)
    return result[None]


# %%
df_train = read_rds(urljoin(BASE_URL_DATA,'data_train_def.rds'))
# df_train.info()
# df_train.describe()


# %%
df_test = read_rds(urljoin(BASE_URL_DATA,'data_test_def.rds'))
# df_test.info()
# df_test.describe()


# %%
ids, df_train = preprocessing(df_train)
# df_train.head()
# df_train.info()


# %%
test_size = round(len(df_test) / (len(df_train)+len(df_test)), 2)

print(f'Test {test_size*100}%')

X_train, X_test, y_train, y_test = split_data(df_train, test_size=test_size)


# %%
pipe = make_transformer_pipeline()
X_train = pipe.fit_transform(X_train).toarray()
y_train = y_train.values


# %%
# X_train, y_train


# %%
X_test = pipe.fit_transform(X_test).toarray()
y_test = y_test.values


# %%
# X_test, y_test



# %%
EPOCHS = 500
BUFFER_SIZE = 1024
BATCH_SIZE = 32
STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE
N_FOLDS = 5

initial_learning_rate = 0.01

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate,
  decay_steps=STEPS_PER_EPOCH*50,
  decay_rate=1,
  staircase=False)


# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=15000,
#     decay_rate=0.95,
#     staircase=False)

# tfa.optimizers.AdamW(learning_rate=0.01, weight_decay=1e-4)

print('BATCH_SIZE: ', BATCH_SIZE)


# %%
train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(buffer_size=BUFFER_SIZE).repeat().batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

train_ds.element_spec


# %%
step = np.linspace(0,STEPS_PER_EPOCH*EPOCHS)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')


# %%
def create_tuner_model(hp):
  # initializer = tf.keras.initializers.he_uniform() 

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(units=hp.Int('units',
                                        min_value=16,
                                        max_value=54,
                                        step=8), activation='relu', input_dim=X_train.shape[1]))
  model.add(tf.keras.layers.Dense(1))
  model.compile(loss='mean_squared_error', 
                optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model

# %%
# tuner = RandomSearch(
#     create_tuner_model,
#     objective='val_loss',
#     max_trials=5,
#     executions_per_trial=3,
#     directory='fmap',
#     project_name='prediccion')


tuner = kerastuner.BayesianOptimization(
    create_tuner_model,
    objective=kerastuner.Objective("val_root_mean_squared_error", direction="min"),
    max_trials=5,
    executions_per_trial=3,
    directory='fmap',
    project_name='prediccion')

tuner.search_space_summary()


# %%
tuner.search(train_ds,
             epochs=EPOCHS, 
             steps_per_epoch=STEPS_PER_EPOCH,
             validation_data=test_ds,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)],
             verbose=0
             )


# %%
tuner.results_summary()


# %%
models = tuner.get_best_models(num_models=2)

# %% 
models

# %%
# Parar entrenamiento cuando no mejora MSE
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

# Guardar modelo entrenado
checkpoint = tf.keras.callbacks.ModelCheckpoint('modelo_prediccion_propiedades.hdf5', monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True, mode='min')

# Reducir learning rate cuando no mejora MSE
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.1, patience=20, min_lr=0.0001)

# %% [markdown]
# ## Entrenamiento simple

# %%
# with strategy.scope():
model = create_model()
      
model.summary()


# %%
# model.load_weights('modelo_prediccion_propiedades.hdf5')

model.fit(train_ds, validation_data=test_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[
        tfdocs.modeling.EpochDots(),
        # reduce_lr,
        checkpoint, 
        early_stop,         
      ], verbose=1)


# %%
result = model.evaluate(test_ds)

# %% [markdown]
# ## Entrenamiento con loop k-fold

# %%
# Loop entrenamiento kfold cross-validation
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=5)
rmse_per_fold = []
loss_per_fold = []
train_history = []

fold_no = 1
for train, val in kfold.split(X_train, y_train):
    with strategy.scope():
      model = create_model()

    model.summary()

    # model.load_weights('modelo_prediccion_propiedades.hdf5')

    print('------------------------------------------------------------------------')
    print(f'Entrenando fold {fold_no} ...')

    history = model.fit(X_train[train], y_train[train], 
      batch_size=BATCH_SIZE, 
      epochs=EPOCHS, 
      validation_data=(X_train[val], y_train[val]), 
      steps_per_epoch = STEPS_PER_EPOCH, 
      callbacks=[
        tfdocs.modeling.EpochDots(),
        checkpoint, 
        early_stop, 
        # reduce_lr 
      ], verbose=1)

    train_history.append(history)

    # Evaluar scores
    scores = model.evaluate(X_train[val], y_train[val], verbose=0)
    print(f'Score para fold {fold_no}: {model.metrics_names[0]} de {scores[0]}; {model.metrics_names[1]} de {scores[1]}')
    rmse_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    tf.keras.backend.clear_session()

    fold_no = fold_no + 1


# SCORES #
print('------------------------------------------------------------------------')
print('Score por fold')
for i in range(0, len(rmse_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - RMSE: {rmse_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Score promedio de todos los folds:')
print(f'> RMSE: {np.mean(rmse_per_fold)} (+- {np.std(rmse_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# %% [markdown]
# # Graficar métricas entrenamiento

# %%
for idx, history in enumerate(train_history):
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']

    plt.plot(np.linspace(0, len(train_rmse),  len(train_rmse)), train_rmse,
                color='red', label='Train')

    plt.plot(np.linspace(0, len(val_rmse), len(val_rmse)), val_rmse,
            color='blue', label='Val')

    plt.title(f'RMSE fold {idx}')
    plt.legend()
    plt.grid(True)
    plt.show()


# %%
loss, rmse = model.evaluate(X_test, y_test, verbose=2)
print("Testing set RMSE: {:5.2f}".format(rmse))


# %%
y_pred = model.predict(X_test).flatten()
y_pred


# %%
plot_test_predictions(y_test, y_pred)

# %% [markdown]
# # Preprocesamiento set de test

# %%
df_test = read_rds(urljoin(BASE_URL_DATA,'data_test_def.rds'))
df_test.head()


# %%
df_test.info()


# %%
ids, df_test = preprocessing(df_test, process_outliers=False)


# %%
df_test.head()


# %%
df_test.drop(['precio'], axis=1, inplace=True)
df_test.head()

# %% [markdown]
# # Predicción con red neuronal

# %%
X = pipe.transform(df_test)
X.shape


# %%
y_pred = model.predict(X).flatten()
data = pd.DataFrame(data={'id' : ids.values, 'precio': np.round(y_pred).astype(int) })
data


# %%
data.to_csv('propiedades_prediccion.csv', index=False)


# %%



