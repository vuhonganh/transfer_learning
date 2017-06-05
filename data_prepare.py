from scipy.misc import imread, imresize, imsave
import glob
import os
import numpy as np

# NOTE: run this file ONLY ONCE

# Suppose that we are in imagenet folder
folder_names = [name for name in os.listdir(".") if os.path.isdir(name) and name[0] != '.']


vgg_size = (224, 224)
id = 1

images = []
labels = []

nb_classes = len(folder_names)

size_class = 12

with open("class_1200.txt", mode="w") as fclass:
    fclass.write("id,class_name,nb_images\n")
    with open("signature_1200.txt", mode="w") as f:
        f.write("im_name,id_class,raw_name\n")
        for id_class in range(len(folder_names)):
            n_per_class = 0
            for file_name in glob.glob(folder_names[id_class] + "/*.JPEG"):
                try:
                    img = imread(file_name, mode="RGB")
                    if img.shape[2] == 3:
                        f.write("%05d.JPEG,%d,%s\n" %(id, id_class, file_name))
                        img = imresize(img, vgg_size)
                        name_img = "../vgg_data/%05d.JPEG" % id
                        imsave(name_img, img)
                        images.append(img)
                        cur_label = np.zeros(nb_classes)
                        cur_label[id_class] = 1.0
                        labels.append(cur_label)
                        id += 1
                        n_per_class += 1
                    else:
                        print("%s does not have 3 channels" % file_name)
                except Exception as e:
                    print("Failed to read file %s: got %s" % (file_name, e))
                if n_per_class == size_class:
                    break
            fclass.write("%d,%s,%d\n" % (id_class, folder_names[id_class], n_per_class))

images = np.asarray(images)
labels = np.asarray(labels)

np.savez("mydata_1200.npz", images=images, labels=labels)