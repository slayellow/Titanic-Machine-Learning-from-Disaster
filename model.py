import numpy as np
import pandas as pd
import tensorflow as tf
import data_analysis as da

class Module():
    def __init__(self):
        pass

    def Binary_Classification_Model(self):
        # Input: Input --> 1024 --> 512 --> 256 --> 128 --> 64 --> 32 --> 16 --> Output
        # Loss Func: BinaryCrossEntropy  /  Optimizer: Adam Optimizer
        # Etc: BatchNormalization / LeakyReLU
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
        ])
        simple_adam = tf.keras.optimizers.Adam(lr=0.01)
        model.compile(optimizer=simple_adam, loss=tf.keras.backend.categorical_crossentropy, metrics=['accuracy'])
        return model

    def train(self, model, train, label, epoch, batch_size):
        hist = model.fit(train, label, epochs=epoch, batch_size=batch_size)
        return hist

    def print_train_result(self, train):
        print('## train loss and acc ##')
        print(train.history['loss'])
        print(train.history['acc'])

    def model_predict(self, model, X_test):
        return model.predict(X_test)

    def evaluate_prediction(self, prediction):
        result = []
        for i, predict in enumerate(prediction):
            argmax = np.argmax(predict)
            result.append(argmax)
        return result

    def save_csv(self, result, filename):
        submission = pd.read_csv('gender_submission.csv')
        submission['Survived'] = result
        submission.to_csv(filename+'.csv', index=False)

if __name__=="__main__":
    dt = da.Data('train.csv','test.csv')
    X_train, Y, X_test = dt.Split_Input_Output()
    Y = tf.keras.utils.to_categorical(Y, 2)
    module = Module()
    model = module.Binary_Classification_Model()
    train = module.train(model=model, train=X_train, label=Y, epoch=300, batch_size=1000)
    module.print_train_result(train)
    predictions = module.model_predict(model, X_test)
    # predictions = np.round(predictions)

    result = module.evaluate_prediction(predictions)

    module.save_csv(result, 'result')
