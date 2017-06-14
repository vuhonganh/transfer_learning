import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    # list all data in history
    #print(history.history.keys())

    y1_1 = history['loss']
    y1_2 = history['val_loss']

    fig, axarr = plt.subplots(nrows=2, sharex=True)
    axarr[0].plot(y1_1, label='train')
    axarr[0].plot(y1_2, label='val')
    axarr[0].set_title('loss')
    axarr[0].legend(loc='upper right')

    y2_1 = history['acc']
    y2_2 = history['val_acc']
    axarr[1].set_title('accuracy')
    axarr[1].plot(y2_1, label='train')
    axarr[1].plot(y2_2, label='val')
    axarr[1].legend(loc='lower right')
    # prefix_file_name = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    # plt.savefig(prefix_file_name + ".png")
    # plt.show()


def plot_accuracy(history):
    train_acc = history['acc']
    val_acc = history['val_acc']
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.legend(loc='lower right')
    plt.axhline(0.9, linestyle='--', label='90% base line')
    plt.show()


def compute_accuracy(integer_label, predictions, cur_classes, debug=True):
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


classes_reader = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
                  "banana", "key", "mug", "pear", "orange"]


def test_from_reader_data(x_test, y_test, model, classes=classes_reader):
    predictions = model.predict(x_test, batch_size=48, verbose=1)
    integer_label = np.argmax(y_test, axis=1)
    acc, acc_3 = compute_accuracy(integer_label, predictions, cur_classes=classes, debug=False)
    print("top 1 accuracy = %f" % acc)
    print("top 3 accuracy = %f" % acc_3)


if __name__ == "__main__":
    test = {'acc':[0.4, 0.5, 0.6, 0.9, 0.95], 'val_acc':[0.2, 0.24, 0.5, 0.67, 0.85]}
    plot_accuracy(test)