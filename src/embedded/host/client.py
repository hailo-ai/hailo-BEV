import socket
import struct
import json

def start_client(out_queue, port, host=None) -> None:
    """
    Starts a TCP client that connects to a server, receives data, and places it in a queue.

    Parameters:
    - out_queue (queue.Queue): Queue to store received data for further processing.
      Each item in the queue is a tuple containing the received message and a unique `sample_token`.
    - port (int, optional): Port number to connect to on the server. Defaults to 5555.
    - host (str, optional): Host address of the server to connect to. Defaults to None,
      which resolves to the local host or IP.

    Behavior:
    - The client connects to the specified host and port.
    - Once connected, it enters a loop to receive messages from the server.
    - For each message:
      - Reads a 4-byte header indicating the message length.
      - Continuously receives chunks until the complete message is received.
      - Decodes the message from JSON format and enqueues it in `out_queue`.
    - If the connection is closed by the server, the function exits.

    Exceptions:
    - Catches any exception to print an error message, then closes the client connection.

    Returns:
    - None: The function operates as a client process and does not return any value.
    """
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        print("Start trying")
        client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")
        while True:

            header = client_socket.recv(4)
            message_length = struct.unpack('I', header)[0]
            json_data = b''
            while len(json_data) < message_length:
                chunk = client_socket.recv(message_length - len(json_data))
                if not chunk:
                    print("Connection closed by the server.")
                    return
                json_data += chunk
            message = json.loads(json_data.decode('utf-8'))
            out_queue.put((message, message[0]['sample_token']))

            # Optional: Wait before sending the next message

    except Exception as exception:
        print(f"An error occurred: {exception}")
    finally:
        # Close the connection
        client_socket.close()
        print("Connection closed.")
