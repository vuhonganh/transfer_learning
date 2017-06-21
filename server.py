#!/usr/bin/env python
import socket
import keras
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow

classes_reader = ["apple", "pen", "book", "monitor", "mouse", "wallet", "keyboard",
                  "banana", "key", "mug", "pear", "orange"]

end = 'MyEnd'.encode()

def recv_end(the_socket):
    total_data = []

    while True:
            cur_data = the_socket.recv(8192)
            #print(type(cur_data))
            if end in cur_data:
                total_data.append(cur_data[:cur_data.find(end)])
                break
            total_data.append(cur_data)
            if len(total_data) > 1:
                # check if end_of_data was split
                last_pair = total_data[-2] + total_data[-1]
                if end in last_pair:
                    total_data[-2] = last_pair[:last_pair.find(end)]
                    total_data.pop()
                    break
    return b''.join(total_data)

if __name__ == "__main__":
    EXIT = "exit".encode()

    model = keras.models.load_model('resnet_512/model.h5')
    s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    port = 12347                # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port

    s.listen(5)                 # Now wait for client connection.

    print("Server ready!")

    conn, addr = s.accept()     # Establish connection with client.

    print('Got connection from ', addr)

    while True:
        # data = conn.recv(1024).decode()
        # data = conn.recv(1024)
        data = recv_end(conn)
        if data == EXIT:
            break
        else:
            # print("received path file from client: ", data)
            # print("Display image")
            # print(type(data))
            # img = imread(data)
            # imshow(img)
            data = np.fromstring(data, dtype=np.uint8)
            print(data.shape)

            img = data.reshape(224, 224, 3)
            # img = imresize(data, (224, 224))
            x = np.asarray([img])
            res = classes_reader[np.argmax(model.predict(x, batch_size=1))]
            # res = 'asdf'
            # imshow(data)
            # time.sleep(2)
            conn.sendall(res.encode())
    conn.sendall(b'bye')
    conn.close()
