import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

import os

base_model_dict = {"resnet": keras.applications.ResNet50,
                   "inception": keras.applications.InceptionV3,
                   "vgg": keras.applications.VGG16,
                   "xception": keras.applications.Xception}

fine_tune_dict = {"resnet": -33,
                  "vgg": -4,
                  "inception": -62,
                  "xception": -15}

if K.image_data_format() == 'channels_first':
    default_input_shape = (3, 224, 224)
else:
    default_input_shape = (224, 224, 3)


def get_fc(base_model, list_hidden_dropout, initialier='he_normal', reg=1e-2, num_classes=12, verbose=True):
    """
    build a fc on top of base model
    :param base_model: conv-model used to transfer learning (vgg, resnet, etc.)
    :param list_hidden_dropout: a list of (hidden_size, dropout_keep_prob)
    :param initialier: how to initialize weights
    :param reg: l2 regularizer factors for weights and biases
    :param num_classes: number of classes to classify
    :param verbose: print out initializer and regularizer detail
    :return: fc model
    """
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))

    if verbose:
        print("initializer = %s and l2 regularizer reg = %f" % (initialier, reg))
    # add hidden layers
    for hidden_size, do_keep_prob in list_hidden_dropout:
        fc_model.add(Dense(hidden_size, activation='relu', kernel_initializer=initialier,
                           kernel_regularizer=keras.regularizers.l2(reg)))
        fc_model.add(Dropout(do_keep_prob))
        if verbose:
            print("added fc-%d with dropout keep probability %f" % (hidden_size, do_keep_prob))

    # do softmax at final layer
    fc_model.add(Dense(num_classes))
    fc_model.add(Activation('softmax'))
    return fc_model


def get_base_model(base_model_name="resnet", fine_tune=False, input_shape=default_input_shape, verbose=True):
    """
    return base model (the conv-net) with or without fine-tune (depends on user's params) 
    :param base_model_name: the conv-net model
    :param fine_tune: if set to True then set last conv-net of base model to be trainable
    :return: base_model
    """
    if base_model_name not in base_model_dict:
        print("unknown model name, use resnet as default")
        base_model_name = "resnet"
    base_model = base_model_dict[base_model_name](include_top=False, weights='imagenet', input_shape=input_shape)
    if fine_tune:
        if verbose:
            print("fine tune %d final layers in conv-net of %s" % (-fine_tune_dict[base_model_name],
                                                                   base_model_name))
        for layer in base_model.layers[:fine_tune_dict[base_model_name]]:
            layer.trainable = False
    else:
        for layer in base_model.layers:
            layer.trainable = False

    return base_model


def get_model(list_hidden_dropout, initialier='he_normal', reg=1e-2, num_classes=12,
              base_model_name="resnet", fine_tune=False, input_shape=default_input_shape, verbose=True):
    base_model = get_base_model(base_model_name, fine_tune, input_shape, verbose)
    fc_model = get_fc(base_model, list_hidden_dropout, initialier, reg, num_classes, verbose)
    model = keras.models.Model(input=base_model.input, output=fc_model(base_model.output))
    return model


def get_model_from_file(whole_model_file_path):
    if os.path.isfile(whole_model_file_path):
        print("load model from file %s" % whole_model_file_path)
        model = keras.models.load_model(whole_model_file_path)
        return model
    else:
        print("No model found at %s" % whole_model_file_path)
        raise ValueError


def build_model(model, learning_rate=1e-4):
    """
    create Adam optimizer and compile model
    :param model:  
    :param learning_rate:
    :return: compiled model 
    """
    adam_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])
    return model


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
    # prefix_file_name = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

    plt.subplot(122)
    #plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(prefix_file_name + ".png")
    plt.show()