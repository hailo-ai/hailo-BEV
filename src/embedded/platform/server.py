import socket
import time
import json
import struct
import queue

def start_server(in_queue, demo, port, host=None) -> None:
    """
    Starts a TCP server to listen for incoming connections and send data from a queue to a client.

    Parameters:
    - in_queue (queue.Queue): Queue containing data to be sent to the connected client.
    - demo (object): Object containing control methods `get_terminate()` and `set_terminate()`,
      used to manage the server's running state.
    - port (int, optional): Port number to bind the server to. Defaults to 5554.
    - host (str, optional): Host address to bind the server to. Defaults to None,
      which binds to all available network interfaces.

    Behavior:
    - The server waits for 10 seconds before initializing.
    - Once started, it creates a socket, binds to the specified host and port,
      and listens for connections.
    - When a client connects, the server enters a loop, where it retrieves data from `in_queue`
      and sends it to the client until `demo.get_terminate()` returns True.
    - For each message, the length of the data (as a header) is sent first,
      followed by the actual data in JSON format.
    - If `get_terminate()` returns True during the loop,
      the connection is closed and the server stops.

    Exceptions:
    - General exceptions are caught to avoid server crashes,
      and the client connection is safely closed 
      in the `finally` block.
    - After closing the connection, `set_terminate()` is called to indicate the server has stopped.

    Returns:
    - None: The function operates as a server process and does not return any value.

    """
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the address and port
    server_socket.bind((host, port))
    print(f"Listening on {host}:{port}")

    # Start listening for incoming connections
    server_socket.listen(5)

    # Accept a new connection
    client_socket, client_address = server_socket.accept()
    print(f"Accepted connection from {client_address}")

    try:
        while True:
            while not demo.get_terminate():
                try:
                    data, _ = in_queue.get(block=True, timeout=1)
                    break
                except queue.Empty:
                    pass
            if demo.get_terminate():
                client_socket.close()
                break
            json_data = json.dumps(data)
            json_bytes = json_data.encode('utf-8')

            # Send the length of the message first
            header = struct.pack('I', len(json_bytes))
            # Send a response to the client
            client_socket.sendall(header + json_bytes)
            if demo.get_terminate():
                client_socket.close()
                break
    except Exception as _:
        pass
    finally:
        # Close the client connection
        client_socket.close()
        demo.set_terminate()
