import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask import Flask, render_template
from fastai.vision.all import *
from pathlib import Path
from fastai.vision.all import *
from fastai.vision.all import load_learner
from PIL import Image


def get_x(r):
    return r['models/pathological_model_3810_v1.pkl']


def get_y(r):
    return r['Finding Labels']


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = Path('models/pathological_model_3810_v1.pkl')
learn_inf = load_learner(model_path)

# model = pickle.load(open('models/pathological_model_3810_v1.pkl', 'rb'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'xray_image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['xray_image']

    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Run your model on the uploaded image
        img_path = Path('uploads/')
        img = Image.open(img_path / filename)
        img.show()

        preds, _, probs = learn_inf.predict(img)

        prediction = "Code hasn't saved"
        for i, (label, prob) in enumerate(zip(learn_inf.dls.vocab, probs)):
            prediction += f"{label}: {prob:.4f}"

        # Replace this line with the actual prediction
        # prediction = "Your model's prediction"
        print(prediction)

        # Data
        labels = learn_inf.dls.vocab
        sizes = [prob.item() for prob in probs]

        # Filter data to include only values above
        filtered_labels = [label for label, size in zip(labels, sizes) if size < 0.3]
        filtered_sizes = [size for size in sizes if size < 0.3]

        return jsonify({'result': prediction})

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
