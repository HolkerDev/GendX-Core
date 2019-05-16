# %% Imports

import codecs
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras import regularizers
from keras.preprocessing import image
from keras.models import load_model
from PIL import JpegImagePlugin
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import max_norm

# %% Initialize classifier

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(64, 64, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation("sigmoid"))

# compile cnn
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %% Callbacks
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
saving_best_loss = ModelCheckpoint(
    'drop_save/20.loss.epo-{epoch:02d}-v_acc-{val_acc:.4f}-v_loss-{val_loss:.4f}.h5',
    monitor='val_loss',
    save_best_only=True, verbose=1)

saving_best_acc = ModelCheckpoint(
    'drop_save/20.acc.epo-{epoch:02d}-v_acc-{val_acc:.4f}-v_loss-{val_loss:.4f}.h5',
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
history = model.fit_generator(
    training_set,
    steps_per_epoch=100,
    epochs=100,
    validation_data=test_set,
    callbacks=[saving_best_loss, saving_best_acc],
    validation_steps=10)

# %% Save trained model
model.save('model_save_last.h5')

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
saveHist('history/20_history.json', history)

# %% Load best trained model
model = load_model('saveback/weights.06-0.35.hdf5')

# %% Show plots for accuracy
matplotlib.use('Qt5Agg')
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
