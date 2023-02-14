from flask import Flask, render_template, Response
from camera import *
from camera import VideoCamera
from faceRecogination import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen1(camera):
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    classFile = "coco.names"
    camera.readClasses(classFile)
    camera.downloadModel(modelURL)
    camera.loadModel()
    while True:
        frame = camera.predictVideo()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2(faceRecogination):
    while True:
        frame = faceRecogination.FaceReco()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_feed')
def video_feed():
    return Response(gen1(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face')
def face():
    return Response(gen2(faceRecogination()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/setting')
def setting():
    return render_template('setting.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True)
