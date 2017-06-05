import numpy as np
import csv

# class_file = "class.txt"
# class_reader = csv.reader(open(class_file))
# for row in class_reader:
#     print(row[2])


def get_data():
    file = np.load("mydata.npz")
    images = file['images']
    labels = file['labels']

    print(images.shape)
    print(labels.shape)

    length = labels.shape[0]

    indices = np.random.permutation(length)

    len_train_val = int(0.8 * length)

    train_val_idx = indices[:len_train_val]
    test_idx = indices[len_train_val:]

    len_train = int(0.8 * len_train_val)
    train_idx = train_val_idx[:len_train]
    val_idx = train_val_idx[len_train:]
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
    get_fc_layers()