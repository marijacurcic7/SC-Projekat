from models import *
import numpy as np
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def instrument_training(features_df, features_gray):

    X = np.array(features_df.feature.tolist())
    yinst = np.array(features_df.instrument.tolist())

    leinst = LabelEncoder()
    yyinst = leinst.fit_transform(yinst)

    x_train, x_test, y_train, y_test = train_test_split(X, yyinst, test_size=0.2, shuffle=True, random_state=42)
    x_validation = x_train.reshape(len(x_train), 128, 44, 1)
    print(x_validation.shape)

    x_test = x_test.reshape(len(x_test), 128, 44, 1)
    print(x_test.shape)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=42)
    x_val = x_val.reshape(len(x_val), 128, 44, 1)
    print(x_val)
    print(x_test)

    num_classes = len(leinst.classes_)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)


    num_epochs = 60
    batch_size = 32
    model = model1(num_classes)

    checkpointer = ModelCheckpoint(filepath="best_weights_inst.hdf5", verbose=1, save_best_only=True)

    start = datetime.now()
    model.fit(x_validation, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val),
              callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Instrument training completed in: ", duration)

    train_score = model.evaluate(x_validation, y_train, verbose=1)
    print("Instrument Training Accuracy: ", train_score[1])

    test_score = model.evaluate(x_test, y_test, verbose=1)
    print("Instrument Test Accuracy: ", test_score[1])

    #prediction
    predictions = model.predict([x_test])
    n = random.randint(0, len(y_test))
    i = np.argmax(predictions[n])
    print(x_test[n])
    print("Prediction: ", leinst.classes_[i])

    for f in features_gray:
        X = np.array(f[1].tolist())
        X = X.reshape(128, 44, 1)

        comparison = X == x_test[n]
        equal_arrays = comparison.all()
        if (equal_arrays):
            print("Class: ", f[2])

