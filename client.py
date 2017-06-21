#!/usr/bin/env python

# This is client.py file
from scipy.misc import imread, imresize, imsave, imshow

import socket               # Import socket module

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 12347                # Reserve a port for your service.

s.connect((host, port))
end = 'MyEnd'

while True:
    print("Enter input (type exit to quit): ", end='')
    in_msg = input()
    # s.sendall(in_msg.encode())
    # suppose msg is file path, read this file and send bytes
    if in_msg != "exit":
        msg = imread(in_msg)
        msg = imresize(msg, (224, 224))
        msg = msg.tostring()
    else:  #
        msg = in_msg.encode()

    s.sendall(msg + end.encode())

    reply_from_server = s.recv(1024).decode()
    print("server predicts: %s" % reply_from_server)
    if reply_from_server == "bye":
        break
