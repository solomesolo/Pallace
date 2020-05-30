from flask import Flask, jsonify, request, render_template, redirect

from werkzeug.utils import secure_filename
import time
from datetime import datetime
import io
from PIL import Image

from predict import *
from preprocess import preprocess_image

UPLOAD_FOLDER = 'uploaded_images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER #"/mnt/c/wsl/projects/pythonise/tutorials/flask_series/app/app/static/img/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 5 * 1024 * 1024

def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
#             if "filesize" in request.cookies:
                
#             if not allowed_image_filesize(request.cookies["filesize"]):
#                 print("Filesize exceeded maximum limit")
#                 return redirect(request.url)

            image = request.files["image"]

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                # create filename with datetime
                filename = secure_filename(image.filename)
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = filename.split('.')[0] + '_' + now +'.' + filename.split('.')[1]
                file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                image.save(file_path)
                print("Image saved")
                return predict(file_path)
                
#                 prediction = predict(file_path)
#                 return render_template('public/result.html')

#                 return redirect(request.url)
            else:
                print("That file extension is not allowed")
                return redirect(request.url)
        
    return render_template("public/upload_image.html")


@app.route('/predict', methods=['POST'])
def predict(file_path):
    print("-------- predict function")
    if request.method == 'POST':
#         file = request.files['file']
#         img_bytes = file.read()
#         image = Image.open(io.BytesIO(image_bytes))
        abnormality_score = get_prediction(file_path=file_path)
        return jsonify({'Abnormality_score': str(abnormality_score), 'Is_abnormal': str(1 if abnormality_score>=0.5 else 0)})    
    
    
    

# @app.route('/upload')
# def upload_file():
#     return render_template('upload.html')

# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save(secure_filename(f.filename))
#         return 'file uploaded successfully'
    
    
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit a empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
if __name__ == '__main__':
    app.run()#debug=True)