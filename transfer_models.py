import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import numpy as np

import matplotlib
from matplotlib.ticker import FuncFormatter
from data_reader import *

base_model_dict = {"resnet": keras.applications.ResNet50,
                   "inception": keras.applications.InceptionV3,
                   "vgg": keras.applications.VGG16,
                   "xception": keras.applications.Xception}

# number of layers until last feature conv-block
fine_tune_dict = {"resnet": 142,
                  "vgg": 15,
                  "inception": 249,
                  "xception": 117}

classes_reader = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
                  "banana", "key", "mug", "pear", "orange"]


class TransferModel:
    def __init__(self, base_model_name, hidden_list,lr=1e-4, num_classes=12, dropout_list=None,
                 reg_list=None, init='he_normal', verbose=True, input_shape=(224, 224, 3)):

        if base_model_name not in base_model_dict:
            print("unknown model name, use resnet as default")
            base_model_name = "resnet"

        self.base_model_name = base_model_name
        # get base model and freeze all layers
        self.base_model = get_base_model(self.base_model_name, input_shape)
        # get fc model
        self.fc_model = get_fc(hidden_list, self.base_model, num_classes, dropout_list, reg_list, init, verbose)
        self.lr = lr
        self.model = build_classification_model(self.base_model, self.fc_model, self.lr)

        self.path_model = create_path_model(self.base_model_name, hidden_list)
        os.makedirs(self.path_model, exist_ok=True)
        self.loss = {'train': [], 'val': []}
        self.acc = {'train': [], 'val': []}
        self.histograms = []
        self.histograms_epochs = []
        self.test_acc = 0.0
        self.testtop_acc = 0.0
        self.cur_epoch = 0
        self.model_name = get_string_from_arr(self.base_model_name, hidden_list, sep='_')

    def set_lr(self, new_lr):
        if new_lr is not None:
            self.lr = new_lr
        else:
            self.lr /= 10.0
        print("set learning rate and recompile model")
        self.model = compiled_model(self.model, self.lr)

    def set_fine_tune(self, new_lr=None):
        """
        set final layers to trainable and reset learning rate         
        """
        # unfreeze final layers
        for layer in self.model.layers[fine_tune_dict[self.base_model_name]:]:
            layer.trainable = True
        # set new lr and recompile model:
        self.set_lr(new_lr)

    def load(self, model_file_path=None):
        # first load model, i.e. load the architecture
        if model_file_path is None:
            model_file_path = self.path_model + "model.h5"
        self.model = keras.models.load_model(model_file_path)

    def fit(self, x_train, y_train, x_val, y_val, bs=96, epos=25, verbose=2, use_early_stop=True, use_lr_reduce=True):
        callBackList = get_call_backs(use_early_stop, use_lr_reduce)
        history = self.model.fit(x_train, y_train, batch_size=bs, epochs=epos, verbose=verbose,
                                 validation_data=(x_val, y_val), callbacks=callBackList)
        self.update_from_history(history, epos)

    def fit_aug(self, x_train, y_train, x_val, y_val, bs=96, epos=30, verbose=2):
        callBackList = get_call_backs()
        train_gen, val_gen = get_gen(x_train)
        history = self.model.fit_generator(train_gen.flow(x_train, y_train, batch_size=bs),
                                           steps_per_epoch=len(x_train) / bs, epochs=epos,
                                           validation_data=val_gen.flow(x_val, y_val, batch_size=bs),
                                           validation_steps=len(x_val) / bs,
                                           verbose=verbose, callbacks=callBackList)
        self.update_from_history(history, epos)

    def update_from_history(self, history, epos):
        self.cur_epoch += epos
        self.loss['train'] += history.history['loss']
        self.loss['val'] += history.history['val_loss']
        self.acc['train'] += history.history['acc']
        self.acc['val'] += history.history['val_acc']

    def evaluate(self, x_test, y_test, top=2, classes=classes_reader):
        predictions = self.model.predict(x_test, batch_size=48, verbose=1)
        integer_label = np.argmax(y_test, axis=1)

        cnt = 0
        cnt_top = 0
        # case prob = 1.0 will belong to 11-th bin: 1.0 <= prob < 1.1
        nb_bin = 11
        hist_range = 1.0 / (nb_bin - 1)
        histo = []
        test_fail = []  # wrong cases
        assignments = np.zeros((y_test.shape[1], y_test.shape[1]))
        f = open(self.path_model + "wrong_prediction.txt", mode='w')
        f.write("predict,truth,prob\n")
        for i in range(predictions.shape[0]):
            preds = np.argsort(predictions[i])[::-1][0:top]
            for p in preds:
                if p == integer_label[i]:
                    cnt_top += 1
                for k in range(nb_bin):
                    if k * hist_range <= predictions[i][p] < (k+1) * hist_range:
                        histo.append((k + 0.5) * hist_range)
            if preds[0] == integer_label[i]:
                cnt += 1
            else:
                test_fail.append(i)
                f.write("%s,%s,%f\n" % (classes[preds[0]], classes[integer_label[i]], predictions[i, preds[0]]))
            assignments[integer_label[i], preds[0]] += 1

        f.close()
        # print assignment tables
        assignments /= np.sum(assignments, axis=1)
        print_assignments(assignments, classes)

        # print top accuracies
        acc = cnt / predictions.shape[0]
        acc_top = cnt_top / predictions.shape[0]
        self.test_acc = acc
        self.testtop_acc = acc_top
        self.histograms.append(histo)
        self.histograms_epochs.append(self.cur_epoch)
        print("top 1 accuracy = %f" % acc)
        print("top %d accuracy = %f" % (top, acc_top))
        return test_fail

    def plot_acc(self, baseline=0.9, savefig=False):
        # clear plot first
        plt.clf()
        plt.plot(self.acc['train'], label='train', color='blue')
        plt.plot(self.acc['val'], label='val', color='red')
        bl_str = '%.2f baseline' % baseline
        plt.plot([0, len(self.acc['train'])], [baseline, baseline], color='black',
                 linestyle='--', label=bl_str)
        plt.xlabel('epoch')
        plt.title('accuracy of %s' % self.model_name)
        plt.legend(loc='lower right')
        if savefig:
            plt.savefig(self.path_model + 'acc.png')
        else:
            plt.show(block=False)

    def plot_loss(self, savefig=False):
        # clear plot first
        plt.clf()
        plt.plot(self.loss['train'], label='train', color='blue')
        plt.plot(self.loss['val'], label='val', color='red')
        plt.xlabel('epoch')
        plt.title('loss of %s' % self.model_name)
        plt.legend(loc='upper right')
        if savefig:
            plt.savefig(self.path_model + 'loss.png')
        else:
            plt.show(block=False)

    def save(self, path=None):
        if path is None:
            path = self.path_model
        os.makedirs(path, exist_ok=True)
        with open(path + "params.txt", mode='w') as f:
            cur_date_time = strftime("date %Y_%m_%d_%H_%M_%S\n", gmtime())
            cur_lr = "lr %f\n" % self.lr
            cur_train_loss = get_string_from_arr("train_loss", self.loss['train'])
            cur_val_loss = get_string_from_arr("val_loss", self.loss['val'])
            cur_train_acc = get_string_from_arr("train_acc", self.acc['train'])
            cur_vall_acc = get_string_from_arr("val_acc", self.acc['val'])
            f.write(cur_date_time + '\n')
            f.write(cur_lr + '\n')
            f.write(cur_train_loss + '\n')
            f.write(cur_val_loss + '\n')
            f.write(cur_train_acc + '\n')
            f.write(cur_vall_acc + '\n')
            if self.test_acc > 0.0:
                f.write("test_acc %f\n" % self.test_acc)
            if self.testtop_acc > 0.0:
                f.write("test_topacc %f\n" % self.testtop_acc)
            np.save(self.path_model + "histograms", np.asarray(self.histograms))

        self.model.save(self.path_model + 'model.h5')
        self.plot_loss(savefig=True)
        self.plot_acc(savefig=True)
        self.plot_histograms(savefig=True)

    def plot_histograms(self, bins=np.linspace(0, 1, 10), savefig=False):
        if len(self.histograms_epochs) == 0:
            return
        names = ['%d epochs' % i for i in self.histograms_epochs]
        # clear plot first
        plt.clf()
        plt.hist(self.histograms, bins=bins, normed=True, label=names)

        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)
        plt.xlim([-0.1, 1.4])
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.legend(loc='upper right')
        plt.title('top2 histogram of %s' % self.model_name)
        if savefig:
            plt.savefig(self.path_model + 'hist.png')
        else:
            plt.show(block=False)

    def plot(self, l=1, a=1, h=1):
        if l:
            self.plot_loss()
        if a:
            self.plot_acc()
        if h:
            self.plot_histograms()


def get_string_from_arr(first_word, arr, sep=' '):
    res = first_word
    for a in arr:
        res += sep + str(a)
    return res


def create_path_model(base_model_name, hidden_list):
    res = base_model_name
    for h in hidden_list:
        res += '_' + str(h)
    res += '/'
    return res


def get_fc(hidden_list, base_model, num_classes=12, dropout_list=None, reg_list=None, init='he_normal', verbose=True):
    """build a fc on top of base model"""
    fc_model = Sequential()    
    fc_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    
    #fc_model.add(AveragePooling2D(pool_size=(7, 7), strides=(7, 7)))  - incompatible, need to be added before fc

    len_hidden = len(hidden_list)
    if verbose:
        print("initializer = %s" % init)
    # add hidden layers
    if dropout_list is None:
        dropout_list = [0.5] * len_hidden
    if reg_list is None:
        reg_list = [5e-3] * len_hidden

    for i in range(len_hidden):
        fc_model.add(Dense(hidden_list[i], activation='relu', kernel_initializer=init,
                                kernel_regularizer=keras.regularizers.l2(reg_list[i])))
        fc_model.add(Dropout(dropout_list[i]))
        if verbose:
            print("added fc-%d, dropout keep prob %f, l2 reg %f" % (hidden_list[i], dropout_list[i], reg_list[i]))
    # do softmax at final layer
    fc_model.add(Dense(num_classes))
    fc_model.add(Activation('softmax'))
    return fc_model


def get_base_model(base_model_name, input_shape=(224, 224, 3)):
    """
    return base model without top, freeze all conv-layers
    """
    base_model = base_model_dict[base_model_name](include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    return base_model


def get_model_from_file(whole_model_file_path):
    try:
        print("load model from file %s" % whole_model_file_path)
        model = keras.models.load_model(whole_model_file_path)
        return model
    except FileNotFoundError:
        print("No model found at %s" % whole_model_file_path)


def build_classification_model(base_model, fc_model, learning_rate=1e-4):
    """
    build classification model = base+fc, use Adam optimizer  
    """
    model = keras.models.Model(inputs=base_model.input, outputs=fc_model(base_model.output))
    return compiled_model(model, learning_rate)


def compiled_model(model, learning_rate):
    adam_opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam_opt,
                  metrics=['accuracy'])
    return model


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(10 * y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def compare_models(list_models, list_names=None):
    plt.clf()
    max_test_acc = 0.0
    max_test_model_name = ''
    for i in range(len(list_models)):
        m = list_models[i]
        if list_names is not None:
            plt.plot(m.acc['val'], label=list_names[i])
        else:
            plt.plot(m.acc['val'], label=m.model_name)
        if m.test_acc > max_test_acc:
            if list_names is not None:
                max_test_model_name = list_names[i]
            else:
                max_test_model_name = m.model_name
            max_test_acc = m.test_acc
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.show(block=False)
    print("best model %s with test accuracy %f" % (max_test_model_name, max_test_acc))


def get_gen(x_train):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=20,
        horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen.fit(x_train)
    return train_datagen, val_datagen


def get_gen_2(x_train):
    train_mean, train_std = get_mean_std(x_train)
    train_datagen = ImageDataGenerator(
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.01,
        zoom_range=0.01,
        rotation_range=2,
        horizontal_flip=True)
    val_datagen = ImageDataGenerator()

    train_datagen.fit(x_train)
    return train_datagen, val_datagen, train_mean, train_std


def get_mean_std(x_train):
    train_mean = np.mean(x_train, axis=0, keepdims=True)
    train_std = np.std(x_train, axis=0, keepdims=True)
    
    # have to make them float32 before putting to CNN
    train_mean = train_mean.astype(np.float32)
    train_std = train_std.astype(np.float32)
    #print("train mean =", train_mean)
    #print("train std =", train_std)
    return train_mean, train_std


def get_call_backs(use_early_stop=True, use_lr_reduce=True):
    cb_list = []
    if use_early_stop:
        earlyStopCallBack = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5,
                                                          verbose=1, mode='auto')
        cb_list.append(earlyStopCallBack)

    if use_lr_reduce:
        lrPlatCallBack = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5,
                                                           verbose=1, mode='auto',
                                                           epsilon=0.0001, cooldown=0, min_lr=5e-6)
        cb_list.append(lrPlatCallBack)
    return cb_list


def exp_2(base_name, hidden_list, augment, use_noise, bs=48, model=None, lr=1e-4, epo1=20, epo2=20, reg_list=None,
          prep=False, normalized=False, verbose=2, use_early_stop=True, use_lr_reduce=True, fine_tune=True, dropout_list=None):
    # load model
    if model is None:
        model = TransferModel(base_name, hidden_list, lr=lr, reg_list=reg_list, dropout_list=dropout_list)

    # get data
    x_train, y_train, x_val, y_val, x_test, y_test = get_data_arr()
    if augment:
        x_train, y_train = get_train_augment(x_train, y_train, use_noise)

    # cast to float32
    x_train = x_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)

    # prepossessing
    if prep:
        train_mean, train_std = get_mean_std(x_train)
        x_train -= train_mean
        x_val -= train_mean
        x_test -= train_mean
        if normalized:
            # x_train /= 255.0
            x_train /= train_std
            # x_val /= 255.0
            x_val /= train_std
            # x_test /= 255.0
            x_test /= train_std

    if epo1 > 0:
        model.fit(x_train, y_train, x_val, y_val, epos=epo1, verbose=verbose, bs=bs,
                  use_early_stop=use_early_stop, use_lr_reduce=use_lr_reduce)
    if fine_tune:
        model.set_fine_tune()
    if epo2 > 0:
        model.fit(x_train, y_train, x_val, y_val, epos=epo2, verbose=verbose, bs=bs,
                  use_early_stop=use_early_stop, use_lr_reduce=use_lr_reduce)

    model.evaluate(x_test, y_test)
    model.plot()
    return model


def do_experiment(base_name, hidden_list, augment, load_model=True, lr=1e-4, epo1=20, epo2=20, reg_list=None, prep=False, verbose=2):
    m = TransferModel(base_name, hidden_list, lr=lr, reg_list=reg_list)
    if load_model:
        m.load()
        m.set_lr(lr)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(augment)

    y_train = keras.utils.to_categorical(y_train, num_classes=12)
    y_val = keras.utils.to_categorical(y_val, num_classes=12)
    y_test = keras.utils.to_categorical(y_test, num_classes=12)
    
    x_train = x_train.astype(np.float32, copy=False)
    x_val = x_val.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)
    
    # IMPORTANT NOTE: for Preprocessing: std does not help, /255 also does not help -> don't know why?
    
    train_mean, train_std = get_mean_std(x_train)
    print(np.min(x_train[0]))
    print(np.min(x_val[0]))
    if prep:
        x_train = (x_train - train_mean)
        x_val = (x_val - train_mean)
        x_test = (x_test - train_mean)
        print(np.min(x_train[0]))
        print(np.min(x_val[0]))
    m.fit(x_train, y_train, x_val, y_val, epos=epo1, verbose=verbose)
    m.set_fine_tune()
    m.fit(x_train, y_train, x_val, y_val, epos=epo2, verbose=verbose)
    m.evaluate(x_test, y_test)
    m.plot()
    return m


def print_assignments(assignments, classes):
    firstline = "%10s" % "class"
    for c in classes:
        firstline += "%10s" % c

    print(firstline)

    for i in range(len(classes)):
        next_line = "%10s" % classes[i]
        for p in assignments[i]:
            next_line += "%10.2f" % p
        print(next_line)

