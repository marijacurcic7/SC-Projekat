from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv3D, BatchNormalization, MaxPool2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def model1(num_classes):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 11), strides=(1, 1), activation='relu', input_shape=(128, 44, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())

    return model