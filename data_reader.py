import numpy as np
from skimage import transform
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import os
from scipy.misc import imread, imresize, imsave, imshow

nb_train = 720
nb_val = 180
nb_test = 300
classes = ['apple', 'banana', 'book', 'key', 'keyboard', 'monitor', 'mouse', 'mug', 'orange', 'pear', 'pen', 'wallet']


def rotate(img, max_degree=20.0):
    return transform.rotate(img, np.random.uniform(-max_degree, max_degree), mode='edge')


def project(img, image_size=224, max_pixel=10.0):
    # random generate 4 corners: top-left, bottom-left, bottom-right, top-right
    tl, bl, br, tr = np.random.uniform(-max_pixel, max_pixel, size=[4, 2])

    bl[1] = image_size - bl[1]
    br = image_size - br
    tr[0] = image_size - tr[0]

    # projection
    trans_obj = transform.ProjectiveTransform()
    trans_obj.estimate(np.array([
        tl,
        bl,
        br,
        tr]), np.array([
        [0, 0],
        [0, image_size],
        [image_size, image_size],
        [image_size, 0]]))

    return transform.warp(img,
                          trans_obj,
                          output_shape=(image_size, image_size),
                          order=1,
                          mode='edge')


def jitter(img):
    img = rotate(img)
    img = project(img)
    img *= 255.0
    return np.ceil(img).astype(np.uint8)


def make_data_set(folder_name, prefix_path="data/", size=(224, 224), augment=False):
    """suppose you have structure: data/train/classes, data/val/classes, data/test/classes"""
    file_name = "%s_aug.npz" % folder_name if augment else "%s.npz" % folder_name
    if os.path.isfile(file_name):
        print("file %s already exists" % file_name)
        return
    print("making %s" % file_name)
    X = []
    y = []
    for i in range(len(classes)):
        class_dir = prefix_path + folder_name + '/' + classes[i] + '/'
        im_names = sorted(os.listdir(class_dir))
        for n in im_names:
            img = imread(class_dir + n, mode='RGB')
            img = imresize(img, size)
            if augment:
                for _ in range(2):
                    X.append(jitter(img))
                    y.append(i)
            X.append(img)
            y.append(i)
    np.savez(file_name, X=np.asarray(X, dtype=np.uint8), y=np.asarray(y, dtype=np.uint8))


def load_data(augment=False):
    dat_name = ["train", "val", "test"]
    fnames = []
    for d in dat_name:
        fn = "%s_aug.npz" % d if augment and d == "train" else "%s.npz" % d
        fnames.append(fn)
        if not os.path.isfile(fn):
            make_data_set(d, augment=augment)
    res = []
    for n in fnames:
        print("loading %s" % n)
        buf = np.load(n)
        res.append(buf['X'])
        res.append(buf['y'])
    return res[0], res[1], res[2], res[3], res[4], res[5]


def test_show_data(idx):
    train = np.load("train.npz")
    x_train = train['X']
    y_train = train['y']
    imshow(x_train[idx])
    print(classes[np.argmax(y_train[idx])])


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


def get_data_arr(file_name="mydata_1200.npz"):
    images, labels, train_idx, val_idx, test_idx = get_data_stratify(file_name)
    x_train = images[train_idx]
    y_train = labels[train_idx]
    x_val = images[val_idx]
    y_val = labels[val_idx]
    x_test = images[test_idx]
    y_test = labels[test_idx]
    return x_train, y_train, x_val, y_val, x_test, y_test


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
    # # get_fc_layers()
    # # images, labels, train_idx, val_idx, test_idx = get_data("mydata_1200.npz")
    # images, labels, train_idx, val_idx, test_idx = get_data_stratify("mydata_1200.npz")
    # integer_labels_test = np.argmax(labels[test_idx], axis=1)
    # integer_labels_train = np.argmax(labels[train_idx], axis=1)
    # integer_labels_val = np.argmax(labels[val_idx], axis=1)
    # integer_labels = np.argmax(labels, axis=1)
    # # plt.hist(integer_labels, bins=12)
    # # plt.hist(integer_labels_test, bins=12)
    # plt.hist(integer_labels_train[64:128], bins=12)
    # plt.hist(integer_labels_train[0:64], bins=12)
    # plt.hist(integer_labels_train[128:192], bins=12)
    # # plt.hist(integer_labels_val, bins=12)
    # plt.show()
    # img = imread('data/train/apple/00002.JPEG', mode='RGB')
    # print(img.shape)
    # imshow(img)
    # img2 = imresize(img, (224, 224))
    # imshow(img2)
    # make_data_set("train")
    # train = np.load("train.npz")
    # x_train = train['X']
    # y_train = train['y']
    # plt.subplot(121)
    # plt.imshow(x_train[5])
    # plt.subplot(122)
    # aug = jitter(x_train[5])
    # plt.imshow(aug)
    # plt.show()

    # img = imread('data/train/apple/00005.JPEG')
    # img = imresize(img, (224, 224))
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(jitter(img))
    # plt.show()
    # val = np.load("val.npz")
    # x_val = val['X']
    # y_val = val['y']

    # imshow(x_val[0])
    # print(classes[np.argmax(y_val[0])])
    # show_data(719)

    make_data_set("train", augment=True)
    make_data_set("val")
    make_data_set("test")