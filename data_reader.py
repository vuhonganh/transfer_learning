import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
# class_file = "class.txt"
# class_reader = csv.reader(open(class_file))
# for row in class_reader:
#     print(row[2])


def get_data(file_name="mydata_1200.npz"):

    np.random.seed(1234)  # VERY IMPORTANT (for running multiple times)

    file = np.load(file_name)
    images = file['images']
    labels = file['labels']

    print(images.shape)
    print(labels.shape)

    length = labels.shape[0]

    indices = np.random.permutation(length)

    len_train_val = int(0.7 * length)

    train_val_idx = indices[:len_train_val]
    test_idx = indices[len_train_val:]
    len_train = int(0.8 * len_train_val)
    train_idx = train_val_idx[:len_train]
    val_idx = train_val_idx[len_train:]
    return images, labels, train_idx, val_idx, test_idx


def get_data_stratify(file_name="mydata_1200.npz"):
    # define spliter
    print("Read data from %s \n" % file_name)
    sss = StratifiedShuffleSplit(test_size=0.3, random_state=1234, n_splits=1)

    # load data
    file = np.load(file_name)
    images = file['images']
    labels = file['labels']
    train_idx, test_idx = None, None
    for tr, te in sss.split(images, labels):
        train_idx = tr
        test_idx = te
    # print(test_idx)

    length_train = int(0.8 * len(train_idx))
    val_idx = train_idx[length_train:]
    train_idx = train_idx[:length_train]

    return images, labels, train_idx, val_idx, test_idx

def get_fc_layers():
    weight_file = "fc_lay.npz"
    weights = np.load(weight_file)
    # keys = sorted(weights.keys())
    keys = weights.keys()
    for i, k in enumerate(keys):
        print(i, k, np.shape(weights[k]))
        # sess.run(self.fc_parameters[i].assign(weights[k]))
    # print(w1.shape)
    # print(b1.shape)
    # print(w1[0])



if __name__ == "__main__":
    # get_fc_layers()
    # images, labels, train_idx, val_idx, test_idx = get_data("mydata_1200.npz")
    images, labels, train_idx, val_idx, test_idx = get_data_stratify("mydata_1200.npz")
    integer_labels_test = np.argmax(labels[test_idx], axis=1)
    integer_labels_train = np.argmax(labels[train_idx], axis=1)
    integer_labels_val = np.argmax(labels[val_idx], axis=1)
    integer_labels = np.argmax(labels, axis=1)
    # plt.hist(integer_labels, bins=12)
    # plt.hist(integer_labels_test, bins=12)
    plt.hist(integer_labels_train[64:128], bins=12)
    plt.hist(integer_labels_train[0:64], bins=12)
    plt.hist(integer_labels_train[128:192], bins=12)
    # plt.hist(integer_labels_val, bins=12)
    plt.show()




