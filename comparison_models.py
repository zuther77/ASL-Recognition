from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16, ResNet50V2, MobileNetV2, InceptionV3


models = {
    'VGG16': VGG16(weights='imagenet',
                   include_top=False, pooling=max, input_shape=(96, 96, 3)),
    'MobileNetV2': MobileNetV2(weights='imagenet',
                               include_top=False, pooling=max, input_shape=(96, 96, 3)),
    'ResNet50V2': ResNet50V2(weights='imagenet',
                             include_top=False, pooling=max, input_shape=(96, 96, 3)),
    'InceptionV3': InceptionV3(weights='imagenet',
                               include_top=False, pooling=max, input_shape=(96, 96, 3))
}


def model_maker(name):
    model = Sequential()
    model.add(models[name])
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(29, activation='softmax'))
    return model
