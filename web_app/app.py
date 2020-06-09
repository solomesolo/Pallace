from flask import Flask, jsonify, request, render_template, redirect, url_for

from werkzeug.utils import secure_filename
import time
from datetime import datetime
import os
import io
from PIL import Image

from predict import *
# from preprocess import preprocess_image

UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

print("__name__:", __name__)
app = Flask(__name__)
current_path = os.path.dirname(os.path.realpath(__file__))
print("current_path:", current_path)
app = Flask(__name__, static_folder=current_path)



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER #"/mnt/c/wsl/projects/pythonise/tutorials/flask_series/app/app/static/img/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 5 * 1024 * 1024
RELATIVE_TEMPLATE_PATH = ''#'../../'


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

@app.route("/", methods=["GET", "POST"])
@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    predicted = False
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
                os.makedirs(app.config["IMAGE_UPLOADS"] +'/'+ filename.split('.')[0])
                file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename.split('.')[0], filename)
                image.save(file_path)
                print("Image saved")
#                 return predict(file_path)
                image, cls_score, heat_map, bounding_box = get_prediction_and_heat_map(file_path)
                probability = round(cls_score*100, 2)
                predicted = True
                # Save images to folder for futher page rendering
                filepath_image_original, filepath_image_bounding_box, filepath_image_heat_map = save_images(filename, image, heat_map, bounding_box)
                # IF NORMAL THEN PLOT ONLY HEAT MAP
                if probability < 50:
                    filepath_image_bounding_box = filepath_image_original
                    message = "Model did not recognize abnormality on x-ray"
                    message_color = 'green'
                else:
                    message = "Model recognized abnormality on x-ray !"
                    message_color = 'red'
#                 print("message:", message)
                return render_template('public/analysis_report_page.html', 
                                        filename                    = filename, 
                                        filepath_image_original     = filepath_image_original, 
                                        filepath_image_bounding_box = filepath_image_bounding_box,
                                        filepath_image_heat_map     = filepath_image_heat_map,
                                        probability                 = probability,
                                        message                     = message,
                                        message_color               = message_color)
                    
#                 return redirect(url_for("analysis_report", filename = filename, 
#                                         filepath_image_original     = filepath_image_original, 
#                                         filepath_image_bounding_box = filepath_image_bounding_box,
#                                         filepath_image_heat_map     = filepath_image_heat_map,
#                                         probability                 = probability))
            else:
                print("That file extension is not allowed")
                return redirect(request.url)
    if not predicted:
        return render_template("public/upload_image.html")
#     if predicted:
#         pass

def save_images(filename, image, heat_map, bounding_box):
    # here draw images with prediction for report web page
    filepath_image_original = filename.split('.')[0]+'_prepr.'+filename.split('.')[1]
    filepath_image_original = os.path.join(app.config["IMAGE_UPLOADS"], filename.split('.')[0], filepath_image_original)
    image_original = Image.fromarray(image)
    image_original.save(filepath_image_original)
    
    # --- Image with bounding box
    image_bounding_box = image.copy()
    # scaling bounding box to 0.5 preserving center point
    x,y,w,h = bounding_box
    x_scaled, y_scaled, w_scaled, h_scaled = (x + int(0.25*w), y + int(0.25*h), x + int(0.75*w), y + int(0.75*h))
    color = (36,255,12) # green
    cv2.rectangle(image_bounding_box, (x_scaled, y_scaled), (w_scaled, h_scaled), color, thickness=3)        
#     print("(x_scaled, y_scaled), (w_scaled, h_scaled):", (x_scaled, y_scaled), (w_scaled, h_scaled))
    # save
    filepath_image_bounding_box = filename.split('.')[0]+'_bb.'+filename.split('.')[1]
    filepath_image_bounding_box = os.path.join(app.config["IMAGE_UPLOADS"], filename.split('.')[0], filepath_image_bounding_box)
    
    image_bounding_box = Image.fromarray(image_bounding_box)
    image_bounding_box.save(filepath_image_bounding_box)
    
    
    # --- Image with heat map overlayed
    image_heat_map = image.copy()
#     heat_map = np.repeat(heat_map[:, :, np.newaxis], 3, axis=2)
#     image_heat_map = image[:,:,0]
    heat_map = (heat_map*255/1.5).astype('uint8')
    
    # Apply colormap
    colormap=cv2.COLORMAP_JET
    heat_map = cv2.applyColorMap(heat_map, colormap)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)

    
#     def rgb(minimum, maximum, value): 
#         minimum, maximum = float(minimum), float(maximum)
#         ratio = 2 * (value-minimum) / (maximum - minimum)
#         b = int(max(0, 255*(1 - ratio)))
#         r = int(max(0, 255*(ratio - 1)))
#         g = 255 - b - r
#         return r, g, b
    
    alpha = 0.5
    image_heat_map = cv2.addWeighted(heat_map, alpha, image_heat_map, 1-alpha, 0)
    # save
    filepath_image_heat_map = filename.split('.')[0]+'_map.'+filename.split('.')[1]
    filepath_image_heat_map = os.path.join(app.config["IMAGE_UPLOADS"], filename.split('.')[0], filepath_image_heat_map)

    image_heat_map = Image.fromarray(image_heat_map)
    image_heat_map.save(filepath_image_heat_map)
    
    
    return filepath_image_original, filepath_image_bounding_box, filepath_image_heat_map
    
  
@app.route("/analysis_report_page/<filename>",  methods=['GET', 'POST'])
def analysis_report(filename, filepath_image_original, filepath_image_bounding_box, filepath_image_heat_map, probability):
    
    if request.method == 'POST':
        print(" RENDERING ANALYSIS")
        return render_template('analysis_report_page.html', filename,  filepath_image_original, filepath_image_bounding_box, filepath_image_heat_map, probability)
    
    
# @app.route("/analysis_report_page/<filename>",  methods=['GET', 'POST'])
# def analysis_report(filename, image, probability, heat_map, bounding_box):
#     print("\n\nFILENAME:", filename)
#     # here draw images with prediction for report web page
#     image_original = image#.copy()
#     filepath_image_original = '/uploaded_images/{}'.format(filename)
#     os.makedirs(filepath_image_original)
#     cv2.imsave(image_original, filepath_image_original)
    
#     image_bounding_box = image.copy()
#     # scaling bounding box to 0.5 preserving center point
#     x,y,w,h = bounding_box
#     x_scaled, y_scaled, w_scaled, h_scaled = (x + int(0.25*w), y + int(0.25*h), x + int(0.75*w), y + int(0.75*h))
#     color = (36,255,12) # green
#     cv2.rectangle(image_bounding_box, (x_scaled, y_scaled), (w_scaled, h_scaled), color, thickness=3)        
    
#     image_heat_map = image.copy()#         return render_template('analysis_report_page.html', filename, filepath_image_original, probability)#, filepath_image_bounding_box, filepath_image_heat_map, probability)

#     image_heat_map = cv2.addWeighted(image_heat_map,0.5,heat_map,0.1,0)
    
#     if request.method == 'POST':
                    

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
    app.run(debug=True)