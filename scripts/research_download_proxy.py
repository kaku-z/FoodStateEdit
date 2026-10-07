"""Temporary loopback CONNECT relay for public research dependencies only.

No request bodies, credentials or URLs are logged. Bind remains loopback and
destinations are restricted to model/code/package hosts. Stop after downloads.
"""
import select
import socket
import socketserver

SUFFIXES=('huggingface.co','hf.co','github.com','githubusercontent.com','pypi.org','pythonhosted.org')
class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        upstream=None
        try:
            line=self.rfile.readline(8192).decode('ascii').strip().split()
            if len(line)!=3 or line[0]!='CONNECT':self.wfile.write(b'HTTP/1.1 405 Method Not Allowed\r\n\r\n');return
            host,port=line[1].rsplit(':',1)
            if int(port)!=443 or not any(host==s or host.endswith('.'+s) for s in SUFFIXES):
                self.wfile.write(b'HTTP/1.1 403 Forbidden\r\n\r\n');return
            while self.rfile.readline(8192) not in (b'\r\n',b'\n',b''):pass
            upstream=socket.create_connection((host,443),timeout=30)
            self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n');self.wfile.flush()
            sockets=[self.connection,upstream]
            while True:
                ready,_,_=select.select(sockets,[],[],90)
                if not ready:break
                for sock in ready:
                    data=sock.recv(262144)
                    if not data:return
                    (upstream if sock is self.connection else self.connection).sendall(data)
        except (OSError,ValueError):pass
        finally:
            if upstream is not None:upstream.close()

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True

if __name__=='__main__':
    with Server(('127.0.0.1',22341),Handler) as s:
        print('Public-dependency relay listening on loopback port 22341',flush=True)
        s.serve_forever()
