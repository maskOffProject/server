from http.server import BaseHTTPRequestHandler
import base64
import generator

def load_binary(filename):
    with open(filename, 'rb') as file_handle:
        return file_handle.read()

def save_image(path, buffer):
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(buffer + b'=' * (-len(buffer) % 4)))

class MOServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        # self.send_header('Content-type','text/html')
        # self.send_header('Content-type', 'image/jpg')
        self.end_headers()  
        binary = load_binary('./result.png')
        self.wfile.write(binary)
        # message = "Hello, World! Here is a GET response"
        # self.wfile.write(bytes(message, "utf8"))
        
    def do_POST(self):
        # Headers
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Read input image
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)

        # Save image locally
        # path = './photo.png'
        # save_image(path, post_body)

        # Generate photo
        res_photo = generator.main(post_body)
        res_photo = res_photo[1]
        # with open(res_path, "rb") as result_image:
        encoded_string = base64.b64encode(res_photo)
 
        # Send response
        self.wfile.write(encoded_string)