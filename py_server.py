import os
import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import json
import urllib.parse as urlparse

from config import get_config
from InferenceManager import InferenceManager


class Resquest(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def getQuery(self):
        print(self.path)
        o = urlparse.urlparse(self.path)
        q = urlparse.parse_qs(o.query)
        upload_file =q['uploadedFilename'][0]
        gallery_file = q['galleryFilename'][0]
        result_file = q['result_file'][0]
        return upload_file, gallery_file, result_file
        # return q['uploadedFilename'][0], q['galleryFilename'][0], q['result_file'][0]

    def do_GET(self):

        upload_filename, galleryFilename, result_path = self.getQuery()
        img_base64 = inferenceManager.infer(upload_filename,
                                            galleryFilename,
                                            result_path)
        print('>>>>>> File exist?: ', os.path.exists(result_path))
        self._set_headers()
        data = {'result_base64': img_base64.decode("utf-8")}  # deco:bytes2str
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        self._set_headers()
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))

        # self.send_response(200)
        # self.end_headers()

        data = json.loads(self.data_string)['params']
        print('>>>>>>>> In do_POST: ')
        filesToCompare = data['filesToCompare']
        uploadedFilename = data['uploadedFilename']
        filename2xCos = inferenceManager.inferWithoutPlotting(uploadedFilename,
                                                              filesToCompare)


        # with open("test123456.json", "w") as outfile:
        #     simplejson.dump(data, outfile)
        # print "{}".format(data)
        # f = open("for_presen.py")
        self.wfile.write(json.dumps((filename2xCos)).encode())
        # self.wfile.write(f.read())


if __name__ == "__main__":
    # Define the http server setting
    PORT = 5901
    # Handler = http.server.SimpleHTTPRequestHandler

    # Define the model config
    mdl_name_default = \
        '2019-08-25-14-35_accuracy:0.9931666666666666_step:218349_None.pth'
    exdir = '/home/r07944011/demo/xcos-demo/public/results'
    dataset = 'lfw'
    dataset_name = dataset
    conf = get_config(training=False)
    # Why bs_size can only be the number that divide 6000 well?
    conf.batch_size = 200
    # Initialize the model manager
    inferenceManager = InferenceManager(conf, mdl_name_default,
                                        exdir, dataset_name)

    # Start the http server
    with socketserver.TCPServer(("", PORT), Resquest) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()
