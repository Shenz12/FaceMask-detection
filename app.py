from flask import Flask, render_template, redirect, url_for
import cv2
from tensorflow.keres.models import load_model
global vs
from detect_mask_video import project


app= Flask(__name__)

prototxtPath="face_detector/deploy.prototxt"
weightsPath="face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)
maskNet=load_model("mask_detector.model")


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/main')
def main():
    return render_template("main.html")

@app.route('/mask')
def emotion():
    project()
    return redirect(url_for('main'))



# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)

