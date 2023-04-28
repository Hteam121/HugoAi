from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask import Flask, render_template
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

model_path = Path('models/path_classifier.pkl')
patho_model = load_learner(model_path)

model_path = Path('models/gender_classifier.pkl')
gender_model = load_learner(model_path)

model_path = Path('models/ageClassifier.pkl')
age_model = load_learner(model_path)

model_path = Path('models/view_classifier.pkl')
viewing_model = load_learner(model_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
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

        # Run your models on the uploaded image
        img_path = Path('uploads/')
        img = Image.open(img_path / filename).convert('RGB')

        # Pathological condition prediction
        patho_preds, _, patho_probs = patho_model.predict(img)
        patho_probs_dict = {label: float(prob) for label, prob in zip(patho_model.dls.vocab, patho_probs)}
        most_probable_condition = max(patho_probs_dict, key=patho_probs_dict.get)

        max_index = np.argmax(patho_probs)
        patho_prediction = patho_model.dls.vocab[max_index]

        for i, (label, prob) in enumerate(zip(patho_model.dls.vocab, patho_probs)):
            print(f"{label}: {prob:.4f}")

        # Load the CSV file
        csv_path = Path('conditions_descriptions.csv')
        df = pd.read_csv(csv_path)

        # Find the description for the predicted condition
        condition_description = df.loc[df['Conditions'] == most_probable_condition]['Description'].values[0]

        # Gender prediction
        gender_preds, _, gender_probs = gender_model.predict(img)
        gender_prediction = gender_preds[0]

        # Age prediction
        age_preds, _, age_probs = age_model.predict(img)
        age_prediction = age_preds[0]

        # Viewing position prediction
        viewing_preds, _, viewing_probs = viewing_model.predict(img)
        viewing_prediction = viewing_preds[0]

        # Return the results as a JSON object
        return jsonify({
            'result': {
                'condition': most_probable_condition,
                'condition_description': condition_description,
                'gender': gender_prediction,
                'age': age_prediction,
                'viewing_position': viewing_prediction
            }
        })

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)
