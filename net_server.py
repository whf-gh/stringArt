import socket
import select


class TCPServer:
    def __init__(self, data_list, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setblocking(False)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.inputs = [self.server_socket]
        self.client_socket = None
        self.list_orig = data_list.copy()
        self.list_copy = data_list.copy()
        self.state = f"Server started, listening on port:{self.port}"

    def handle_events(self):
        if self.inputs:
            readable, _, exceptional = select.select(self.inputs, [], self.inputs, 0)
            for s in readable:
                if s is self.server_socket:
                    self.client_socket, addr = s.accept()
                    print(f"Client connected from {addr}")
                    self.client_socket.setblocking(False)
                    self.inputs.append(self.client_socket)
                    self.state = f"Client connected from {addr}"
                else:
                    self.handle_client(s)
            for s in exceptional:
                self.inputs.remove(s)
                s.close()

    def handle_client(self, client_socket):
        try:
            request = client_socket.recv(1024)
            if request:
                request = request.rstrip(b"\0")
                num_elements = int(request.decode())
                self.state = f"received request for {num_elements} elements"
                if num_elements == 0:
                    response = str(self.get_data_length())
                else:
                    response = self.get_elements(num_elements)
                response += "\0"
                client_socket.sendall(response.encode())
            else:
                self.inputs.remove(client_socket)
                client_socket.close()
                self.list_copy = self.list_orig.copy()
                self.state = f"Server started, listening on port:{self.port}"
        except Exception as e:
            print(f"Error: {e}")
            # self.inputs.remove(client_socket)
            # client_socket.close()
            self.state = f"Error: {e}"

    def get_elements(self, num_elements):
        if num_elements > len(self.list_copy):
            num_elements = len(self.list_copy)
        elements = self.list_copy[:num_elements]
        self.list_copy = self.list_copy[num_elements:]
        return ",".join(map(str, elements))

    def get_state(self):
        return self.state

    def get_data_length(self):
        return len(self.list_copy)

    def stop(self):
        for s in self.inputs:
            s.close()
            self.state = "Server closed"
        self.inputs = []


# Example usage:
if __name__ == "__main__":
    server = TCPServer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "127.0.0.1", 65432)
    while True:
        server.handle_events()
