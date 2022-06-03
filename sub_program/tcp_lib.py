from socket import *


class tcp_server_com():
    def __init__(self):
        self.client_add = "localhost"
        self.port = 6340
        self.server_socket = None
        self.connection_socket = None

    def connect_to_client(self):
        self.server_socket = socket(AF_INET, SOCK_STREAM)
        self.server_socket.bind((self.client_add, self.port))
        self.server_socket.listen(1)
        print("waiting client connection...")

        self.connection_socket, add = self.server_socket.accept()
        print("connection from", str(add[0]))

    def disconnect(self):
        self.connection_socket.close()
        print("disconnection successful")

    def send_msg(self, message):
        self.connection_socket.send(len(message).to_bytes(4, byteorder='little'))
        self.connection_socket.send(message.encode('utf-8'))

    def receive_msg(self):
        receive_msg_len = int.from_bytes(self.connection_socket.recv(4), byteorder='little')
        receive_msg = self.connection_socket.recv(receive_msg_len).decode('utf-8')

        return receive_msg


class tcp_client_com():
    def __init__(self):
        self.host_add = 'localhost'
        self.client_socket = None
        self.port = 6340

    def connect_to_server(self):
        self.client_socket = socket(AF_INET, SOCK_STREAM)
        self.client_socket.connect((self.host_add, self.port))
        print("successful connected to server of " + self.host_add)

    def disconnect(self):
        self.client_socket.close()
        print("disconnection successful")

    def send_msg(self, message):
        self.client_socket.send(len(message).to_bytes(4, byteorder='little'))
        self.client_socket.send(message.encode('utf-8'))

    def receive_msg(self):
        receive_msg_len = int.from_bytes(self.client_socket.recv(4), byteorder='little')
        receive_msg = self.client_socket.recv(receive_msg_len).decode('utf-8')

        return receive_msg

