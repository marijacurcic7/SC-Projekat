from models import *
import numpy as np
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

def pitch_svm(features_df):

    X = np.array(features_df.feature.tolist())
    y = np.array(features_df.pitch.tolist())

    le = LabelEncoder()
    yyinst = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, yyinst, test_size=0.2, shuffle=True, random_state=42)

    model = SVC(kernel="linear")

    #print(x_train.shape)

    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    print("Train Accuracy:", metrics.accuracy_score(y_train, y_pred_train))

    y_pred_test = model.predict(x_test)
    print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred_test))

    print(set(y_test) - set(y_pred_test))

    print(metrics.classification_report(y_test, y_pred_test, target_names=le.classes_))



def pitch_training(features_df, features_gray):

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
    num_epochs = 60
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


    #preditction
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
            print("Class: ", f[3])

