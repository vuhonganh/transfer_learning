import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

# NOTE THAT classes should be in sorted order because generator KERAS will do so if we don't specify classes
classes = ['apple', 'banana', 'book', 'key', 'keyboard', 'monitor', 'mouse', 'mug', 'orange', 'pear', 'pen', 'wallet']
num_classes = len(classes)
img_width, img_height = 224, 224

train_data_dir = 'data/train'
val_data_dir = 'data/val'
test_data_dir = 'data/test'

data_augmentation = False

NB_TRAIN_PER_CLASS = 720
NB_VAL_PER_CLASS = 180
NB_TEST_PER_CLASS = 300

nb_train_samples = NB_TRAIN_PER_CLASS * num_classes
nb_validation_samples = NB_VAL_PER_CLASS * num_classes
epochs = 20
batch_size = 24

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def vgg16_model():

    base_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    print('model vgg16 loaded without top')
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    fc_model.add(Dense(1024, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(256, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(num_classes))
    fc_model.add(Activation('softmax'))
    fc_model.load_weights('bottleneck_fc_model.h5')

    model = keras.models.Model(input=base_model.input, output=fc_model(base_model.output))

    for layer in model.layers[:25]:
        layer.trainable = False

    adam_opt = keras.optimizers.Adam(lr=1e-5, decay=1e-6)


    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])

    if data_augmentation:
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size
                                                        )
    val_generator = val_datagen.flow_from_directory(val_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size
                                                    )

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=nb_validation_samples//batch_size,
                        )

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = keras.applications.VGG16(include_top=False, weights='imagenet')

    # class_mode = None to make our generator only yield batches of data, no labels
    # this is necessary because predict_generator need this kind of generator (only data)
    generator = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            classes=classes,
                                            class_mode=None,
                                            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size, verbose=1)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # the label is easy to recreate once we don't use shuffle above
    bottleneck_label_train = np.array([i // NB_TRAIN_PER_CLASS for i in range(nb_train_samples)])
    np.save('bottleneck_label_train.npy', bottleneck_label_train)

    generator = datagen.flow_from_directory(val_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            classes=classes,
                                            class_mode=None,
                                            shuffle=False)

    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size, verbose=1)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

    # again, recreating label is easy if we don't use shuffle above
    bottleneck_label_val = np.array([i // NB_VAL_PER_CLASS for i in range(nb_validation_samples)])
    np.save('bottleneck_label_val.npy', bottleneck_label_val)


def add_fc_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_label = keras.utils.to_categorical(np.load('bottleneck_label_train.npy'), num_classes=num_classes)
    val_data = np.load('bottleneck_features_validation.npy')
    val_label = keras.utils.to_categorical(np.load('bottleneck_label_val.npy'), num_classes=num_classes)

    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=train_data.shape[1:]))
    fc_model.add(Dense(1024, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(256, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(num_classes))
    fc_model.add(Activation('softmax'))
    adam_opt = keras.optimizers.Adam(lr=1e-5, decay=1e-6)

    fc_model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    fc_model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_label))
    fc_model.save('bottleneck_fc_model.h5')


def do_on_top():
    if os.path.isfile('bottleneck_features_train.npy'):
        add_fc_model()
    else:
        save_bottlebeck_features()


vgg16_model()
