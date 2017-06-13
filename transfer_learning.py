import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
from data_reader import get_data_stratify
import matplotlib.pyplot as plt

from time import gmtime, strftime

# NOTE THAT classes should be in sorted order because generator KERAS will do so if we don't specify classes
classes = ['apple', 'banana', 'book', 'key', 'keyboard', 'monitor', 'mouse', 'mug', 'orange', 'pear', 'pen', 'wallet']
num_classes = len(classes)
img_width, img_height = 224, 224

train_data_dir = 'data/train'
val_data_dir = 'data/val'
test_data_dir = 'data/test'

data_augmentation = True

NB_TRAIN_PER_CLASS = 720
NB_VAL_PER_CLASS = 180
NB_TEST_PER_CLASS = 300

nb_train_samples = NB_TRAIN_PER_CLASS * num_classes
nb_validation_samples = NB_VAL_PER_CLASS * num_classes
nb_test_samples = NB_TEST_PER_CLASS * num_classes
epochs = 1
batch_size = 48

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def get_fc_model_1(base_model):
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    #fc_model.add(Dropout(0.5))
    fc_model.add(Dense(1024, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.7))
    fc_model.add(Dense(256, activation='relu', kernel_initializer='VarianceScaling'))
    fc_model.add(Dropout(0.5))
    fc_model.add(Dense(num_classes))
    fc_model.add(Activation('softmax'))
    return fc_model


def get_vgg_old():
    # TODO: should we use pooling?
    base_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # base_model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    print('model vgg16 loaded without top')
    fc_model = get_fc_model_1(base_model)
    #fc_model.load_weights('bottleneck_fc_model.h5')  # in case fine-tuning
    model = keras.models.Model(input=base_model.input, output=fc_model(base_model.output))
    return model


def get_resnet():
    base_model = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    # base_model = keras.applications.ResNet50(include_top=False, weights='imagenet',
    #                                          input_shape=input_shape, pooling='max')
    for layer in base_model.layers:
        print("freeze layer", layer)
        layer.trainable = False
    fc_model = get_fc_model_1(base_model)
    model = keras.models.Model(input=base_model.input, output=fc_model(base_model.output))
    return model


def get_gen_from_dir():
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
    return train_generator, val_generator


def test_from_dir(model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      classes=classes,
                                                      class_mode=None,
                                                      shuffle=False)
    predictions = model.predict_generator(test_generator, nb_test_samples // batch_size, verbose=1)
    np.save('predictions.npy', predictions)
    test_label = [i // NB_TEST_PER_CLASS for i in range(nb_test_samples)]
    cnt = 0
    cnt_top_3 = 0
    for i in range(predictions.shape[0]):
        print("\ncurrent item %d: " % i)
        print("expect class %s" % classes[test_label[i]])
        preds = (np.argsort(predictions[i])[::-1])[0:3]
        for p in preds:
            print(classes[p], predictions[i][p])
            if p == test_label[i]:
                cnt_top_3 += 1

        if preds[0] == test_label[i]:
            cnt += 1

    print("\n top 1 test accuracy = %f" % (cnt / nb_test_samples))
    print("\n top 3 test accuracy = %f" % (cnt_top_3 / nb_test_samples))


def train_vgg16_model_from_dir(nb_epoch=1, learning_rate=1e-4, cur_batch_size=batch_size):
    model = get_vgg_old()
    # freeze until the last conv block (conv 5) to fine tune this last one
    # print(model.layers)
    for layer in model.layers[:15]:
        print("freeze layer", layer)
        layer.trainable = False

    adam_opt = keras.optimizers.Adam(lr=learning_rate)

    # model.load_weights('old_model_keras.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])

    train_generator, val_generator = get_gen_from_dir()

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // cur_batch_size,
                        epochs=nb_epoch,
                        validation_data=val_generator,
                        validation_steps=nb_validation_samples//cur_batch_size,
                        )
    model.save('old_model_keras.h5')


def compute_accuracy(integer_label, predictions, cur_classes=classes, debug=True):
    """
    calculate top-1 and top-3 accuracy
    :param integer_label: 1D array contains index of class
    :param predictions: prediction probabilities correspondingly
    :return: top-1 and top-3 accuracy
    """
    cnt = 0
    cnt_top_3 = 0
    for i in range(predictions.shape[0]):
        if debug:
            print("\ncurrent item %d: " % i)
            print("expect class %s" % cur_classes[integer_label[i]])
        preds = np.argsort(predictions[i])[::-1][0:3]
        for p in preds:
            if debug:
                print(cur_classes[p], predictions[i][p])
            if p == integer_label[i]:
                cnt_top_3 += 1
        if preds[0] == integer_label[i]:
            cnt += 1
    return cnt / predictions.shape[0], cnt_top_3 / predictions.shape[0]


def test_from_reader_data(x_test, y_test, model):
    predictions = model.predict(x_test, batch_size=64, verbose=1)
    integer_label = np.argmax(y_test, axis=1)
    classes_reader = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
                      "banana", "key", "mug", "pear", "orange"]
    acc, acc_3 = compute_accuracy(integer_label, predictions, cur_classes=classes_reader, debug=False)
    print("top 1 accuracy = %f" % acc)
    print("top 3 accuracy = %f" % acc_3)


def train_vgg_from_reader(nb_epoch=1, learning_rate=1e-4, cur_batch_size=64, continue_training=True):
    images, labels, train_idx, val_idx, test_idx = get_data_stratify()
    print("done loading data")
    model = get_vgg_old()
    print("done loading model")

    if continue_training:
        print("continue training from h5 model file")
        model.load_weights('reader_model_keras.h5')

    # freeze all conv net
    for layer in model.layers[:19]:
        print("freeze layer", layer)
        layer.trainable = False

    adam_opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])

    x_train = images[train_idx]
    y_train = labels[train_idx]
    x_val = images[val_idx]
    y_val = labels[val_idx]
    x_test = images[test_idx]
    y_test = labels[test_idx]

    history = model.fit(x_train, y_train, batch_size=cur_batch_size, epochs=nb_epoch, validation_data=(x_val, y_val),
                        verbose=1)
    print("saving model")
    model.save('reader_model_keras.h5')
    print("model saved!\n")

    print("testing model")
    test_from_reader_data(x_test, y_test, model)

    print("plotting training process")
    plot_history(history)


def plot_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(1)
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    prefix_file_name = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

    plt.subplot(122)
    #plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(prefix_file_name + ".png")
    #plt.show()

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
    fc_model.add(Dropout(0.5))
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


def input_hyperparams():
    print("enter number of epoch: ", end='')
    try:
        nb_epochs = int(input())
    except ValueError:
        print("got error, use default nb epoch then")
        nb_epochs = 1

    print("enter learning rate: ", end='')
    try:
        learning_rate = float(input())
    except ValueError:
        print("got error, use default learning rate then")
        learning_rate = 1e-4

    print("enter batch_size: ", end='')
    try:
        user_batch_size = int(input())
    except ValueError:
        print("got error, use default learning rate then")
        user_batch_size = 64
    return nb_epochs, learning_rate, user_batch_size


def train_resnet_from_reader(nb_epoch=1, learning_rate=1e-4, cur_batch_size=64):
    images, labels, train_idx, val_idx, test_idx = get_data_stratify()
    print("done loading data")
    model = get_resnet()
    print("done loading model resnet")

    adam_opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])

    x_train = images[train_idx]
    y_train = labels[train_idx]
    x_val = images[val_idx]
    y_val = labels[val_idx]
    x_test = images[test_idx]
    y_test = labels[test_idx]

    history = model.fit(x_train, y_train, batch_size=cur_batch_size, epochs=nb_epoch, validation_data=(x_val, y_val),
                        verbose=1)
    print("saving model")
    model.save('reader_resnet_keras.h5')
    print("model saved!\n")

    print("testing model")
    test_from_reader_data(x_test, y_test, model)

    print("plotting training process")
    plot_history(history)


nb_epochs, learning_rate, user_batch_size = input_hyperparams()
# # train_vgg16_model_from_dir(nb_epochs)
print("enter model to train (0:vgg, 1:resnet): ", end='')
try:
    choose_resnet = int(input())
except ValueError:
    choose_resnet = 0
    print("got error, default is vgg")

if choose_resnet:
    train_resnet_from_reader(nb_epochs, learning_rate, user_batch_size)
else:
    train_vgg_from_reader(nb_epochs, learning_rate, user_batch_size)
