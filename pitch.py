from models import *
import numpy as np
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def pitch_training(features_df):

    X = np.array(features_df.feature.tolist())
    yinst = np.array(features_df.pitch.tolist())

    leinst = LabelEncoder()
    yyinst = leinst.fit_transform(yinst)

    x_train, x_test, y_train, y_test = train_test_split(X, yyinst, test_size=0.2, shuffle=True, random_state=42)
    x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=42)

    x_train = x_train.reshape(len(x_train), 128, 44, 1)
    x_test = x_test.reshape(len(x_test), 128, 44, 1)
    x_validation = x_validation.reshape(len(x_validation), 128, 44, 1)
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)

    num_classes = len(leinst.classes_)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_validation = to_categorical(y_validation, num_classes=num_classes)

    # training
    num_epochs = 100
    batch_size = 32
    model = model1(num_classes)

    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", verbose=1, save_best_only=True)

    start = datetime.now()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_validation, y_validation), callbacks=[checkpointer], verbose=1)

    duration = datetime.now() - start
    print("Pitch training completed in: ", duration)

    train_score = model.evaluate(x_train, y_train, verbose=1)
    print("Pitch Training Accuracy: ", train_score[1])

    # test
    test_score = model.evaluate(x_test, y_test, verbose=1)
    print("Pitch Test Accuracy: ", test_score[1])

    # predictions = model.predict([x_test])
    # n = random.randint(0, len(y_test))
    # i = np.argmax(predictions[n])
    # print(leinst.classes_[i])
    # print(x_test[n])

