''' A simple Neural Network for binary classification on Titanic Survival.'''
from keras.layers import Dense, Dropout
from keras import Sequential
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

# Load your data, sailor
training = pd.read_csv('./train_preprocessed.csv')
print(training.head())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_train = training[features].values
Y_train = training['Survived'].values.reshape(len(training), 1) # Column Vector

input_shape = X_train.shape[1]

def build_model(input_shape):
    model = Sequential()

    model.add(Dense(256*2, activation='relu', input_dim=input_shape, kernel_initializer='glorot_uniform'))
    model.add(Dense(128*2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(64*2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(32*2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(16*2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(8*2, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


model = build_model(input_shape)
history = model.fit(X_train, Y_train, validation_split=0.3, epochs=600, batch_size=64, verbose=2)

model.save('./model/titanic_model.h5')

### Plot results.
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()




