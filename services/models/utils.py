import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import feature_column
import pandas_profiling as pdpf

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

adapt_data = np.array([1, 2, 3, 4, 5], dtype='float32')

"""
data = pd.read_csv('./services/models/train.csv')

# media edad
mean_value = round(data['Age'].mean())
# valor mas frecuente
mode_value = data['Embarked'].mode()[0]
value = {'Age': mean_value, 'Embarked': mode_value}
data.fillna(value=value, inplace=True)
# Elimina las filas con nulos
data.dropna(axis=1,inplace=True)

train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Survived')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch )

def get_scal(feature):
  def minmax(x):
    mini = train[feature].min()
    maxi = train[feature].max()
    return (x - mini)/(maxi-mini)
  return(minmax)

#numarical features
num_c = ['Age','Fare','Parch','SibSp']
bucket_c = ['Age'] #bucketized numerical feature
#categorical features
cat_i_c = ['Embarked', 'Pclass','Sex'] #indicator columns
cat_e_c = ['Ticket'] # embedding column

feature_columns = []
for header in num_c:
  scal_input_fn = get_scal(header)
  feature_columns.append(feature_column.numeric_column(header, normalizer_fn=scal_input_fn))

for feature_name in cat_i_c:
  vocabulary = data[feature_name].unique()
  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  one_hot = feature_column.indicator_column(cat_c)
  feature_columns.append(one_hot)

for feature_name in cat_e_c:
  vocabulary = data[feature_name].unique()
  cat_c = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
  embeding = feature_column.embedding_column(cat_c, dimension=5)
  feature_columns.append(embeding)


# modelo
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
  layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'),
  layers.Dropout(0.1),
  
  layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds,
          validation_data=val_ds,
          epochs=20)

loss, accuracy = model.evaluate(test_ds)
print('Accuracy: ', accuracy)
# Valores unicos de codificacion one-hot
#point = feature_column.categorical_column_with_vocabulary_list('point', df['point'].unique())
#print(point)
#point_emb = feature_column.embedding_column(point, dimension=4)
#demo(point_emb)
"""