#!/usr/bin/env python3
"""SSH ProxyCommand through an HTTP CONNECT proxy (the compute nodes reach the internet only via
$https_proxy, and have no nc / socat):  ProxyCommand=python3 web/proxy_connect.py %h %p"""
import os
import select
import socket
import sys
from urllib.parse import urlparse

host, port = sys.argv[1], sys.argv[2]
p = urlparse(os.environ.get("https_proxy") or os.environ["http_proxy"])
s = socket.create_connection((p.hostname, p.port or 3128), timeout=30)
s.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
head = b""
while b"\r\n\r\n" not in head:
    chunk = s.recv(1)
    if not chunk:
        sys.exit("proxy closed the connection")
    head += chunk
if b" 200" not in head.split(b"\r\n")[0]:
    sys.exit(f"proxy refused: {head.splitlines()[0].decode(errors='replace')}")
s.settimeout(None)
fin, fout = sys.stdin.buffer.raw, sys.stdout.buffer.raw
while True:
    r, _, _ = select.select([s, fin], [], [])
    if s in r:
        d = s.recv(65536)
        if not d:
            break
        fout.write(d)
    if fin in r:
        d = os.read(fin.fileno(), 65536)
        if not d:
            s.shutdown(socket.SHUT_WR)
            continue
        s.sendall(d)
