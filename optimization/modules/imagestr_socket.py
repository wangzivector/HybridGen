import socket
import numpy as np

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
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip_add, sock_id))
            print('Socket connected: ', ip_add, self.mode)
            self.operator = self.sock

        else: raise ValueError('Mode is invaild :{}'.format(self.mode))
     

    def Send(self, image, string):
        img_data = image.astype(np.double).tobytes()
        img_name = string.encode()
        self.operator.send(img_data)
        self.operator.send(img_name)
        # print('{}: Send: imagesize:{} namesize:{}'\
        #     .format(self.mode, len(img_data), len(img_name)))


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
        # print('{}: Rece: imagesize:{} name:{}'\
        #     .format(self.mode, img_numpy.shape, str_name))
        return img_numpy, str_name


    def Close(self):
        self.sock.close()


"""
########### Client Example: ########### 

import time
import numpy as np
from imagestr_socket import ImageStrserverClient

if __name__ == '__main__':
    client = ImageStrserverClient(mode='CLIENT', ip_add='127.0.0.1', sock_id=8004)
    image = np.arange(300*300).reshape((300,300)).astype(np.float64)

    for i in range(10):
        client.Send(image, 'ImageNamec')
        client.Get(720000, 10)

        time.sleep(2)

    client.Close()


########### server Example: ########### 

import numpy as np
import time
from imagestr_socket import ImageStrserverClient

if __name__ == '__main__':
    server = ImageStrserverClient(mode='SERVER', ip_add='127.0.0.1', sock_id=8004)
    image = np.arange(300*300).reshape((300,300)).astype(np.float64)

    for i in range(10):
        i_numpy, i_name = server.Get(720000, 10)
        server.Send(image, 'ImageNames')

        time.sleep(1)

    server.Close()

########### End ########### 
"""

"""
Deplucated Below:
"""

# class ImageStrClient:
#     def __init__(self, ip_add='127.0.0.1', sock_id=8004):
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.connect((ip_add, sock_id))
     
#     def Send(self, frame, name):
#         img_data = frame.tobytes()
#         img_name = name.encode()
#         self.sock.send(img_data)
#         self.sock.send(img_name)
#         print('SERVER: Send: ', 'imagesize:{}, namesize:{}'\
#             .format(len(img_data), len(img_name)))

#     def Get(self, img_size, str_size):
#         img_buff = self.sock.recv(img_size, socket.MSG_WAITALL)
#         str_buff = self.sock.recv(str_size, socket.MSG_WAITALL)
#         img_numpy = np.frombuffer(img_buff)
#         str_name = str_buff.decode()
#         img_numpy = img_numpy.reshape((300,300))
#         print('Client: Receive: ', str_name, img_numpy.shape)
#         return img_numpy, str_name

#     def Close(self):
#         self.sock.close()


# class ImageStrServer:
#     def __init__(self, ip_add='127.0.0.1', sock_id=8004):
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         self.sock.bind((ip_add, sock_id))
#         self.sock.listen(1)
#     # def WaitforConnect(self):
#         print('Waiting Socket connect...')
#         self.conn, self.addr = self.sock.accept()
#         print('Socket connected: ', self.addr)
    
#     def Send(self, frame, name):
#         img_data = frame.tobytes()
#         img_name = name.encode()
#         self.conn.send(img_data)
#         self.conn.send(img_name)
#         print('SERVER: Send: ', 'imagesize:{}, namesize:{}'\
#             .format(len(img_data), len(img_name)))

#     def Get(self, img_size, str_size):
#         img_buff = self.conn.recv(img_size, socket.MSG_WAITALL)
#         str_buff = self.conn.recv(str_size, socket.MSG_WAITALL)
#         img_numpy = np.frombuffer(img_buff)
#         str_name = str_buff.decode()
#         img_numpy = img_numpy.reshape((300,300))
#         print('SERVER: Receive: ', str_name, img_numpy.shape)
#         return img_numpy, str_name

#     def Close(self):
#         self.sock.close()

# if __name__ == '__main__':
#     image_server = ImageStrServer()
#     image = np.arange(300*300).reshape((300,300)).astype(np.float64)

#     for i in range(10):
#         i_numpy, i_name = image_server.Get(720000, 10)
#         image_server.Send(image, 'ImageNames')

#         time.sleep(1)

#     image_server.Close()