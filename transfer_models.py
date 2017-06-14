import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import numpy as np

import matplotlib
from matplotlib.ticker import FuncFormatter

base_model_dict = {"resnet": keras.applications.ResNet50,
                   "inception": keras.applications.InceptionV3,
                   "vgg": keras.applications.VGG16,
                   "xception": keras.applications.Xception}

# number of layers until last feature conv-block
fine_tune_dict = {"resnet": 142,
                  "vgg": 15,
                  "inception": 249,
                  "xception": 117}

class TransferModel:
    def __init__(self, base_model_name, hidden_list, lr=1e-4, num_classes=12, dropout_list=None,
                 reg_list=None, init='he_normal', verbose=True, input_shape=(224, 224, 3)):

        if base_model_name not in base_model_dict:
            print("unknown model name, use resnet as default")
            base_model_name = "resnet"

        self.base_model_name = base_model_name
        # get base model and freeze all layers
        self.base_model = get_base_model(self.base_model_name, input_shape)
        # number of layers in base model
        self.nb_conv_layers = len(self.base_model.layers)
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
        self.test2_acc = 0.0
        self.cur_epoch = 0

    def set_lr(self, new_lr):
        if new_lr is not None:
            self.lr = new_lr
        else:
            self.lr /= 10.0

    def set_fine_tune(self, new_lr=None):
        """
        set final layers to trainable and reset learning rate         
        """
        self.set_lr(new_lr)
        # unfreeze final layers
        for layer in self.base_model.layers[fine_tune_dict[self.base_model_name]:]:
            layer.trainable = True
        # rebuild model:
        self.model = build_classification_model(self.base_model, self.fc_model, self.lr)

    def set_model_from_file(self, model_file_path):
        self.model = get_model_from_file(model_file_path)

    def fit(self, x_train, y_train, x_val, y_val, bs=48, epos=25, verbose=2):
        lrPlatCallBack = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=3,
                                                           verbose=1, mode='auto',
                                                           epsilon=0.0001, cooldown=0, min_lr=1e-6)

        history = self.model.fit(x_train, y_train, batch_size=bs, epochs=epos, verbose=verbose,
                                 validation_data=(x_val, y_val), callbacks=[lrPlatCallBack])
        self.cur_epoch += epos
        self.loss['train'] += history.history['loss']
        self.loss['val'] += history.history['val_loss']
        self.acc['train'] += history.history['acc']
        self.acc['val'] += history.history['val_acc']

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test, batch_size=48, verbose=1)
        integer_label = np.argmax(y_test, axis=1)
        # classes_reader = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
        #                   "banana", "key", "mug", "pear", "orange"]
        cnt = 0
        cnt_top_2 = 0
        nb_bin = 10
        hist_range = 1.0 / nb_bin
        histo = []

        for i in range(predictions.shape[0]):
            preds = np.argsort(predictions[i])[::-1][0:2]
            for p in preds:
                if p == integer_label[i]:
                    cnt_top_2 += 1
                for k in range(nb_bin):
                    if k * hist_range <= predictions[i][p] < (k+1) * hist_range:
                        histo.append((k + 0.5) * hist_range)
            if preds[0] == integer_label[i]:
                cnt += 1
        acc = cnt / predictions.shape[0]
        acc_2 = cnt_top_2 / predictions.shape[0]
        self.test_acc = acc
        self.test2_acc = acc_2
        self.histograms.append(histo)
        self.histograms_epochs.append(self.cur_epoch)
        print("top 1 accuracy = %f" % acc)
        print("top 2 accuracy = %f" % acc_2)

    def plot_acc(self, baseline=0.9, savefig=False):
        plt.plot(self.acc['train'], label='train', color='blue')
        plt.plot(self.acc['val'], label='val', color='red')
        bl_str = '%.2f baseline' % baseline
        plt.plot([0, len(self.acc['train'])], [baseline, baseline], color='black',
                 linestyle='--', label=bl_str)
        plt.xlabel('epoch')
        plt.title('accuracy')
        plt.legend(loc='lower right')
        if savefig:
            plt.savefig(self.path_model + 'acc.png')
        else:
            plt.show(block=False)

    def plot_loss(self, savefig=False):
        plt.plot(self.loss['train'], label='train', color='blue')
        plt.plot(self.loss['val'], label='val', color='red')
        plt.xlabel('epoch')
        plt.title('loss')
        plt.legend(loc='upper right')
        if savefig:
            plt.savefig(self.path_model + 'loss.png')
        else:
            plt.show(block=False)

    def save_model(self):
        with open(self.path_model + "params.txt", mode='w') as f:
            cur_date_time = strftime("date %Y_%m_%d_%H_%M_%S\n", gmtime())
            cur_lr = "lr %f\n" % self.lr
            cur_train_loss = get_string_from_arr("train_loss", self.loss['train'])
            cur_val_loss = get_string_from_arr("val_loss", self.loss['val'])
            cur_train_acc = get_string_from_arr("train_acc", self.acc['train'])
            cur_vall_acc = get_string_from_arr("val_acc", self.acc['val'])
            f.write(cur_date_time)
            f.write(cur_lr)
            f.write(cur_train_loss)
            f.write(cur_val_loss)
            f.write(cur_train_acc)
            f.write(cur_vall_acc)
            if self.test_acc > 0.0:
                f.write("test_acc %f\n" % self.test_acc)
            if self.test2_acc > 0.0:
                f.write("test_2acc %f\n" % self.test2_acc)
            np.save(self.path_model + "histograms", np.asarray(self.histograms))

        self.model.save(self.path_model + 'model.h5')
        self.plot_loss(savefig=True)
        self.plot_acc(savefig=True)

    def plot_histograms(self, bins=np.linspace(0, 1, 10), savefig=False):
        names = ['%d epochs' % (i + 1) for i in range(len(self.histograms))]
        plt.hist(self.histograms, bins=bins, normed=True, label=names)

        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)
        plt.xlim([-0.1, 1.4])
        # Set the formatter
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.legend(loc='upper right')
        plt.title('histogram of top 2 accuracies by number of epochs')
        if savefig:
            plt.savefig(self.path_model + 'hist.png')
        else:
            plt.show(block=False)


def get_string_from_arr(first_word, arr):
    res = first_word
    for a in arr:
        res += ' ' + str(a)
    res += '\n'
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
