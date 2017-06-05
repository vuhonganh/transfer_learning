import numpy as np


# CURRENT RESULT: test accuracy is 0.886806
# final error at each batch in range 5-20

classes = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
           "banana", "key", "mug", "pear", "orange"]

ass_tab = np.load("assignments_table.npy")

for i in range(12):
    print("%8s" % classes[i], end=' ')
    print()
    for i in range(12):
        for j in range(12):
            print("%8.2f " % ass_tab[i, j], end='')
    print()


