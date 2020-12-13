from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

import sys
import traceback

import time
import os
from ObjectRecognizer import ObjectRecognizer


# ssh jhong12@128.8.235.4
# sudo lsof -i -P -n | grep LISTEN
# nvidia-smi
# source ./venv/bin/activate  
# run virtual env
# nohup python3 -u TOR_HTTP_Server_v3.py &> server_log &
# ps -aux
# sudo ps -U jhong12

# ssh -N -f -L localhost:4000:localhost:4000 jhong12@128.8.235.4

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self):
        self.isTraining = False
        self.object_recognizer = ObjectRecognizer()
        self.object_recognizer.debug = True

    def safeGetValue(self, dic, k):
        if k in dic:
            return dic[k]
        else:
            return None

    def parseParams(self):
        content_length = int(self.headers['Content-Length'])
        html_body = self.rfile.read(content_length)

        postStr = html_body.decode().replace('%2F', '/')
        postStr = postStr.split('&')
        params = {}
        for pstr in postStr:
            pstr = pstr.split('=')
            params[pstr[0]] = pstr[1]

        print(params)
        userID = self.safeGetValue(params, 'userID')
        cmd = self.safeGetValue(params, 'type')
        category = self.safeGetValue(params, 'category')
        img_path = self.safeGetValue(params, 'imgPath')
        return userID, cmd, category, img_path

    def do_GET(self):
        print('GET')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        try:
            start_time = time.time()
            userID, cmd, category, img_path = self.parseParams()
            model_dir = '/home/jhong12/TOR-app-files/models/' + userID
            response = BytesIO()

            if cmd == 'test':
                best_label, entropy, conf = self.object_recognizer.predict(model_dir, img_path)

                output = ''
                if best_label is None:
                    output = 'Object recognition model does not exist.'
                else:
                    output = str(entropy)
                    for label, confidence in conf.items():
                        output = output + '-:-' + label + '/' + str(confidence)

                response.write(str.encode(output))

            elif cmd == 'loadModel':
                self.object_recognizer.loadModelAndLabels(model_dir)

            elif cmd == 'trainRequest':
                markFile = '/home/jhong12/TOR-app-files/isTraining'
                f = open(markFile, 'w')
                f.write('yes')
                f.close()
                train_img_dir = '/home/jhong12/TOR-app-files/photo/TrainFiles/' + userID + '/' + category

                self.object_recognizer.saveModelAndLabel(model_dir+'_prev', org_dir=model_dir)
                self.object_recognizer.train(model_dir, train_img_dir)

                f = open(markFile, 'w')
                f.write('no')
                f.close()

                print('print response: training is done.')
                response.write(b'Training is done')

            else:
                print('Debugging...')

            self.wfile.write(response.getvalue())
        except:
            print('Exception. Reset training.')
            markFile = '/home/jhong12/TOR-app-files/isTraining'
            f = open(markFile, 'w')
            f.write('no')
            f.close()

            print(traceback.format_exc())

            e = sys.exc_info()[0]
            msg = 'Error: %s' % e
            self.wfile.write(msg.encode())

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # second gpu

print('run')
markFile = '/home/jhong12/TOR-app-files/isTraining'
f = open(markFile, 'w')
f.write('no')
f.close()

httpd = HTTPServer(('128.8.235.4', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
