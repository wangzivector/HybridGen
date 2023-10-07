#!/usr/bin/env python

import time
import numpy as np
import socket


"""
Socket Node
"""
class ImageStrServerClient:
    def __init__(self, mode='SERVER/CLIENT', ip_add='127.0.0.1', sock_id=3456):
        self.mode = mode

        if self.mode == 'SERVER':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((ip_add, sock_id))
            self.sock.listen(1)
            print('Waiting Socket connect...')
            self.conn, self.addr = self.sock.accept()
            print('Socket connected: ', self.addr, self.mode)
            self.operator = self.conn

        elif self.mode == 'CLIENT':
            is_client_activate = False
            while not is_client_activate:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.connect((ip_add, sock_id))
                    is_client_activate = True
                except:
                    print('Waiting Socket server...{}'.format(ip_add))
                    is_client_activate = False
                    time.sleep(1)
            print('Socket connected: ', ip_add, self.mode)
            self.operator = self.sock

        else: raise ValueError('Mode is invaild :{}'.format(self.mode))
     

    def Send(self, image, string):
        img_data = image.astype(np.double).tobytes()
        img_name = string.encode()
        self.operator.send(img_data)
        self.operator.send(img_name)
        print('{}: Send: imagesize:{} namesize:{}'\
            .format(self.mode, len(img_data), len(img_name)))


    def Get(self, img_size, str_size, shape=(-1, 300,300)):
        # peername = self.sock.getpeername() # check if still available
        img_buff = self.operator.recv(img_size, socket.MSG_WAITALL)
        str_buff = self.operator.recv(str_size, socket.MSG_WAITALL)
        if len(img_buff) == 0: 
            self.Close()
            raise RuntimeError('socket is disconnected.')
        img_numpy = np.frombuffer(img_buff)
        str_name = str_buff.decode()
        img_numpy = img_numpy.reshape(shape).astype(np.single)
        print('{}: Rece: imagesize:{} name:{}'\
            .format(self.mode, img_numpy.shape, str_name))
        return img_numpy, str_name

    def Close(self):
        self.sock.close()
