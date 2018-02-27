import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

data = pd.read_pickle("new_features_mapped.pkl")

for col in data:
    print(col)
print("Loading dataset")
X = np.array(data["feature_vector"].tolist())
Y = np.load("Train_one_hot_new.pkl")
print("Loaded dataset")
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
print("Splitted the dataset")
X = X.reshape(-1,len(X))
print(X_train.shape)
print(X_train[0].shape)
model = Sequential()
model.add(Dense(units=1024, activation='relu', input_dim=X.shape[0]))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(units=48, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
print("Starting training")

model.fit(X_train, y_train, epochs=20, batch_size=1000)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=1000)

print(loss_and_metrics)
