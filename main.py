# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

# %%
csv_file = 'test.csv'
dataframe = pd.read_csv(csv_file)

# %%
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# %%
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('result')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 62 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


feature_columns = []

age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[20,25,30,35,40,45,60,70])
feature_columns.append(age_buckets)

unit = feature_column.categorical_column_with_vocabulary_list('unit', ['guoqi', 'siqi', 'geti'])
unit_one_hot = feature_column.indicator_column(unit)
feature_columns.append(unit_one_hot)

xueli = feature_column.categorical_column_with_vocabulary_list('xueli', ['gaozhong', 'zhuanke', 'benke', 'shuoshi', 'boshi'])
xueli_one_hot = feature_column.indicator_column(xueli)
feature_columns.append(xueli_one_hot)

hunyin = feature_column.categorical_column_with_vocabulary_list('hunyin', ['weihun', 'yihun', 'lihun', 'sang_ou'])
hunyin_one_hot = feature_column.indicator_column(hunyin)
feature_columns.append(hunyin_one_hot)

zhiye = feature_column.categorical_column_with_vocabulary_list('zhiye', ['guanli', 'jiaoxue', 'jishu', 'nongmin', 'sale', 'siji', 'weixiu', 'wenyuan', 'wuye'])
zhiye_one_hot = feature_column.indicator_column(zhiye)
feature_columns.append(zhiye_one_hot)

sex = feature_column.categorical_column_with_vocabulary_list('sex', ['man', 'woman'])
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

feature_columns.append(feature_column.numeric_column("hours"))

address = feature_column.categorical_column_with_vocabulary_list('address', ['village', 'city'])
address_one_hot = feature_column.indicator_column(address)
feature_columns.append(address_one_hot)



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(64, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

checkpoint_save_path = 'model.ckpt'  
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)  

# %%
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True, save_best_only=True)
history = model.fit(train_ds,validation_data=val_ds,epochs=50, callbacks=[cp_callback])

# %%

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

# # %%
# sample = {
#     'age': 31,
#     'unit': 'siqi',
#     'xueli': 'zhuanke',
#     'hunyin': 'yihun',
#     'zhiye': 'jishu',
#     'sex': 'man',
#     'hours': 42,
#     'address': 'village'
# }
# input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
# predictions = model.predict(input_dict)
# prob = tf.nn.sigmoid(predictions[0])

# print("%.1f ok." % (100 * prob))
# %%
