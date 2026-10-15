"""Persistent framed TCP, application byte accounting; no pickle RPC."""
import json
import socket
import struct
import zlib

FRAME = struct.Struct("!II")
MAX_FRAME = 512 * 1024 * 1024


def encode_message(meta, payload=b""):
    header = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode()
    return FRAME.pack(len(header), len(payload)) + header + payload


def decode_message(data):
    if len(data) < FRAME.size:
        raise ValueError("Truncated RPC frame")
    a, b = FRAME.unpack_from(data)
    if len(data) != FRAME.size + a + b:
        raise ValueError("RPC frame length mismatch")
    return json.loads(data[FRAME.size:FRAME.size + a]), data[FRAME.size + a:]


def read_exact(sock, n):
    parts = []
    while n:
        part = sock.recv(min(n, 1024 * 1024))
        if not part:
            raise ConnectionError("Peer closed connection")
        parts.append(part)
        n -= len(part)
    return b"".join(parts)


def receive(sock):
    header = read_exact(sock, FRAME.size)
    a, b = FRAME.unpack(header)
    if a + b > MAX_FRAME:
        raise ValueError("RPC frame too large")
    return header + read_exact(sock, a + b)


class TCPClient:
    def __init__(self, host, port, timeout=300):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def exchange(self, message):
        self.sock.sendall(message)
        return receive(self.sock)

    def close(self):
        self.sock.close()


def checked_payload(meta, payload):
    if meta.get("payload_crc32") != zlib.crc32(payload):
        raise ValueError("Response payload checksum mismatch")
    return payload
