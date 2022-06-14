# Python 3 server example
import webserver
from http.server import HTTPServer

hostName = ""
serverPort = 3000


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), webserver.MOServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass