import socket

def request_file(server_ip, port, filename):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    client_socket.sendall(filename.encode())

    with open('received_' + filename, 'wb') as file:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            file.write(data)

    client_socket.close()

if __name__ == "__main__":
    server_ip = '192.168.56.1'  # Remplacez par l'IP du serveur
    port = 12345
    filename = "PyMuPDF-1.24.10-cp312-none-manylinux2014_x86_64.whl"
    request_file(server_ip, port, filename)
