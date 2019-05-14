# %% Imports
import codecs
import json
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.preprocessing import image
from keras.models import load_model
from PIL import JpegImagePlugin
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import max_norm

# %% Initialize classifier
classifier = Sequential()

# convolution layer
classifier.add(Conv2D(32, (3, 3), kernel_constraint=max_norm(3), bias_constraint=max_norm(3), input_shape=(64, 64, 3),
                      activation='relu'))

# pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.5))

# flattening layer
classifier.add(Flatten())

# full connection layer (first dense layer)
classifier.add(Dense(units=128, activation='relu'))

# Second Dense layer
classifier.add(Dense(units=1, activation='sigmoid'))

# compile cnn
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% Callbacks
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
saving_best_h5 = ModelCheckpoint('dropout_saveback_h5/14.weights.epo-{epoch:02d}-acc-{val_acc:.4f}.h5',
                                 monitor='val_acc',
                                 save_best_only=True, verbose=1)

# %% Fit model
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# test data generator configuration
test_datagen = ImageDataGenerator(rescale=1. / 255)

# train dataset
training_set = train_datagen.flow_from_directory(
    # path to training folder
    'dataset/training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# test dataset
test_set = test_datagen.flow_from_directory(
    # path to test folder
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# start training
history = classifier.fit_generator(
    training_set,
    steps_per_epoch=500,
    epochs=60,
    validation_data=test_set,
    callbacks=[saving_best_h5],
    validation_steps=200)

# %% Save trained model
classifier.save('model_save_last.h5')

# %% Save classifier history
with open('history/history_forth.json', 'w') as f:
    json.dump(history.history, f)


# %% Save history function
def saveHist(path, history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)


def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n


# %% Save history
saveHist('history/history_drop.json', history)

# %% Load best trained model
model = load_model('saveback/weights.06-0.35.hdf5')

# %% Show plots for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% Show plots for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %% Check function

def check(path):
    files = image.load_img(path, target_size=(64, 64))

    # converting to array
    test_image = image.img_to_array(files)

    # extend by 1 dimension to (64,64,3)
    test_image = np.expand_dims(test_image, axis=0)

    # getting result
    result = model.predict(test_image)
    if result[0] == 0:
        print('man')
    else:
        print('woman')


# %% Check single image
check('dataset/predict/man-in-suit2.jpg')
