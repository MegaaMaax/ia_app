import socket
import os

def send_file(client_socket, filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            client_socket.sendall(file.read())
    else:
        client_socket.sendall(b"ERROR: File not found")

def start_server(host='0.0.0.0', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f'Server listening on {host}:{port}')

    while True:
        client_socket, addr = server_socket.accept()
        print(f'Connection from {addr}')
        filename = client_socket.recv(1024).decode()
        send_file(client_socket, filename)
        client_socket.close()

if __name__ == "__main__":
    start_server()
