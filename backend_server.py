from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
from werkzeug.utils import secure_filename
import json
import numpy as np
from pathlib import Path
from src.inference.predict import TumorPredictor
from flask import send_file, session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
from sklearn.metrics import roc_curve, auc, confusion_matrix
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import uuid
import sentry_sdk
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'dcm', 'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'models/saved_models/best_model.h5'
ANALYTICS_PATH = 'analytics/analytics.json'
RESULTS_FOLDER = 'static/results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs('analytics', exist_ok=True)

# Sentry integration
import os
SENTRY_DSN = os.environ.get('SENTRY_DSN')
if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)

# Load or initialize analytics
if os.path.exists(ANALYTICS_PATH):
    try:
        with open(ANALYTICS_PATH, 'r') as f:
            analytics_data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        analytics_data = {'total_images': 0, 'tumor_percentages': [], 'recent_cases': []}
        with open(ANALYTICS_PATH, 'w') as f:
            json.dump(analytics_data, f)
else:
    analytics_data = {'total_images': 0, 'tumor_percentages': [], 'recent_cases': []}

# Load model once
predictor = TumorPredictor(MODEL_PATH)

# User setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
bcrypt = Bcrypt(app)
USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        # Create an initial admin user if file does not exist
        admin_id = str(uuid.uuid4())
        admin_user = {'username': 'admin', 'password_hash': bcrypt.generate_password_hash('admin123').decode('utf-8')}
        users = {admin_id: admin_user}
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        print('Created users.json with default admin user (admin/admin123)')
        return users
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    print('Saving users to:', USERS_FILE)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
    print('Users saved:', users)

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash
    @staticmethod
    def get(user_id):
        users = load_users()
        for uid, u in users.items():
            if uid == user_id:
                return User(uid, u['username'], u['password_hash'])
        return None
    @staticmethod
    def get_by_username(username):
        users = load_users()
        for uid, u in users.items():
            if u['username'] == username:
                return User(uid, u['username'], u['password_hash'])
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(f'Registration attempt: username={username}')
        users = load_users()
        if any(u['username'] == username for u in users.values()):
            flash('Username already exists.')
            print('Registration failed: username exists')
            return render_template('register.html')
        user_id = str(uuid.uuid4())
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        users[user_id] = {'username': username, 'password_hash': password_hash}
        try:
            save_users(users)
            flash('Registration successful. Please log in.')
            print('Registration successful')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error saving user: {e}')
            print(f'Error saving user: {e}')
            return render_template('register.html')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(f'Login attempt: username={username}')
        user = User.get_by_username(username)
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Logged in successfully.')
            print('Login successful')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
            print('Login failed: invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

# Utility to generate PDF report
def generate_pdf_report(result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont('Helvetica-Bold', 18)
    c.drawString(50, y, 'Tumor Detection Report')
    y -= 40
    c.setFont('Helvetica', 12)
    c.drawString(50, y, f"Prediction Time: {result.get('prediction_time', '')}")
    y -= 30
    c.drawString(50, y, f"Tumor Percentage: {result.get('tumor_percentage', 0):.2f}%")
    y -= 20
    c.drawString(50, y, f"Tumor Area: {int(result.get('tumor_area', 0))} pixels")
    y -= 20
    c.drawString(50, y, f"Total Area: {int(result.get('total_area', 0))} pixels")
    y -= 30
    c.setFont('Helvetica-Bold', 14)
    c.drawString(50, y, 'Metadata:')
    y -= 20
    c.setFont('Helvetica', 12)
    metadata = result.get('metadata', {})
    for k, v in metadata.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 18
        if y < 100:
            c.showPage()
            y = height - 50
    # Overlay image
    overlay_path = result.get('overlay_path')
    if overlay_path:
        try:
            img_path = os.path.join('static', overlay_path)
            c.showPage()
            c.setFont('Helvetica-Bold', 14)
            c.drawString(50, height - 50, 'Overlay Image:')
            c.drawImage(ImageReader(img_path), 50, height//2 - 100, width=400, height=300, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            c.drawString(50, y, f"[Overlay image could not be loaded: {e}]")
    c.save()
    buffer.seek(0)
    return buffer

@app.route('/download_report')
def download_report():
    result = session.get('last_result')
    if not result:
        flash('No analysis result available for report.')
        return redirect(url_for('index'))
    pdf_buffer = generate_pdf_report(result)
    return send_file(pdf_buffer, as_attachment=True, download_name='tumor_report.pdf', mimetype='application/pdf')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def make_result_session_safe(result):
    # Only keep JSON-serializable fields
    safe = {}
    import numpy as np
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        elif isinstance(v, np.generic):
            safe[k] = v.item()
    # Always include overlay_path, original_image_path, patient_id, patient_name, notes, tumor stats, prediction_time, metadata
    for key in ['overlay_path', 'original_image_path', 'patient_id', 'patient_name', 'notes', 'tumor_percentage', 'tumor_area', 'total_area', 'prediction_time', 'metadata']:
        if key in result:
            v = result[key]
            if isinstance(v, np.generic):
                safe[key] = v.item()
            else:
                safe[key] = v
    return safe

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    import numpy as np  # Defensive import in case np is not in scope
    if request.method == 'POST':
        files = request.files.getlist('file')
        if not files or not files[0].filename:
            flash('No files selected')
            return redirect(request.url)
        patient_id = request.form.get('patient_id', '').strip()
        patient_name = request.form.get('patient_name', '').strip()
        notes = request.form.get('notes', '').strip()
        if not patient_id:
            flash('Patient ID is required.')
            return redirect(request.url)
        results = []
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                try:
                    threshold = float(request.form.get('threshold', 0.5))
                    result = predictor.predict(file_path, threshold)
                    overlay = predictor.create_overlay(result['original_image'], result['prediction'])
                    overlay_filename = f"{Path(filename).stem}_overlay.png"
                    overlay_path = os.path.join(RESULTS_FOLDER, overlay_filename)
                    from matplotlib import pyplot as plt
                    # Normalize and clip overlay to [0, 1], then save as uint8
                    overlay = np.clip(overlay, 0, 1)
                    overlay_uint8 = (overlay * 255).astype(np.uint8)
                    plt.imsave(overlay_path, overlay_uint8)
                    result['overlay_path'] = f"results/{overlay_filename}"
                    original_filename = f"{Path(filename).stem}_original.png"
                    original_path = os.path.join(RESULTS_FOLDER, original_filename)
                    # Robustly handle all original image cases
                    orig_img = result['original_image']
                    # Remove singleton channel if present
                    if orig_img.ndim == 3 and orig_img.shape[2] == 1:
                        orig_img = np.squeeze(orig_img, axis=2)
                    # Remove alpha channel if present
                    if orig_img.ndim == 3 and orig_img.shape[2] == 4:
                        orig_img = orig_img[..., :3]
                    # Convert grayscale to RGB for display
                    if orig_img.ndim == 2:
                        orig_img = np.stack([orig_img]*3, axis=-1)
                    # Replace NaN/Inf with safe values
                    orig_img = np.nan_to_num(orig_img, nan=0.0, posinf=255, neginf=0)
                    # Stretch contrast if dynamic range is too small
                    if np.issubdtype(orig_img.dtype, np.floating):
                        vmin, vmax = np.nanmin(orig_img), np.nanmax(orig_img)
                        if vmax - vmin < 1e-3:
                            orig_img = np.zeros_like(orig_img)
                        else:
                            orig_img = (orig_img - vmin) / (vmax - vmin)
                        orig_img = np.clip(orig_img, 0, 1)
                        orig_img = (orig_img * 255).astype(np.uint8)
                    elif orig_img.dtype != np.uint8:
                        vmin, vmax = np.min(orig_img), np.max(orig_img)
                        if vmax - vmin < 1:
                            orig_img = np.zeros_like(orig_img, dtype=np.uint8)
                        else:
                            orig_img = ((orig_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                    # Resize if very large
                    if orig_img.shape[0] > 1024 or orig_img.shape[1] > 1024:
                        scale = 1024.0 / max(orig_img.shape[0], orig_img.shape[1])
                        new_size = (int(orig_img.shape[1]*scale), int(orig_img.shape[0]*scale))
                        orig_img = np.array(Image.fromarray(orig_img).resize(new_size, Image.Resampling.BILINEAR))
                    print("DEBUG: orig_img shape:", orig_img.shape, "dtype:", orig_img.dtype, "min:", orig_img.min(), "max:", orig_img.max())
                    Image.fromarray(orig_img).save(original_path)
                    result['original_image_path'] = f"results/{original_filename}"
                    result['patient_id'] = patient_id
                    result['patient_name'] = patient_name
                    result['notes'] = notes
                    result['binary_prediction'] = result.get('binary_prediction')
                    result['ground_truth'] = None
                    analytics_data['total_images'] += 1
                    analytics_data['tumor_percentages'].append(result['tumor_percentage'])
                    analytics_data['recent_cases'].append({
                        'filename': filename,
                        'tumor_percentage': result['tumor_percentage'],
                        'time': result['prediction_time'],
                        'patient_id': patient_id,
                        'patient_name': patient_name
                    })
                    analytics_data['recent_cases'] = analytics_data['recent_cases'][-10:]
                    with open(ANALYTICS_PATH, 'w') as f:
                        json.dump(analytics_data, f)
                    results.append(result)
                except Exception as e:
                    flash(f'Analysis failed for {filename}: {e}')
            else:
                flash(f'Invalid file type: {file.filename}')
        if len(results) == 1:
            session['last_result'] = make_result_session_safe(results[0])
            case_history = session.get('case_history', [])
            case_history.append(make_result_session_safe(results[0]))
            session['case_history'] = case_history[-10:]
            session['patient_context'] = {'patient_id': patient_id, 'patient_name': patient_name}
            return render_template('results.html', result=results[0])
        elif len(results) > 1:
            session['batch_results'] = [make_result_session_safe(r) for r in results]
            return render_template('batch_results.html', results=results, patient_id=patient_id, patient_name=patient_name)
        else:
            return redirect(request.url)
    return render_template('index.html')

@app.route('/view_case', methods=['POST'])
@login_required
def view_case():
    case_index = int(request.form.get('case_index', -1))
    case_history = session.get('case_history', [])
    if 0 <= case_index < len(case_history):
        case = case_history[case_index]
        session['last_result'] = case
        # Update patient context for topbar
        session['patient_context'] = {
            'patient_id': case.get('patient_id', ''),
            'patient_name': case.get('patient_name', '')
        }
        return render_template('results.html', result=case)
    flash('Invalid case selected.')
    return redirect(url_for('index'))

@app.route('/view_batch_case', methods=['POST'])
@login_required
def view_batch_case():
    case_index = int(request.form.get('case_index', -1))
    batch_results = session.get('batch_results', [])
    if 0 <= case_index < len(batch_results):
        case = batch_results[case_index]
        session['last_result'] = case
        session['patient_context'] = {
            'patient_id': case.get('patient_id', ''),
            'patient_name': case.get('patient_name', '')
        }
        return render_template('results.html', result=case)
    flash('Invalid case selected.')
    return redirect(url_for('index'))

@app.route('/download_batch_report', methods=['POST'])
@login_required
def download_batch_report():
    case_index = int(request.form.get('case_index', -1))
    batch_results = session.get('batch_results', [])
    if 0 <= case_index < len(batch_results):
        case = batch_results[case_index]
        pdf_buffer = generate_pdf_report(case)
        return send_file(pdf_buffer, as_attachment=True, download_name='tumor_report.pdf', mimetype='application/pdf')
    flash('Invalid case selected.')
    return redirect(url_for('index'))

def compute_roc_and_confusion(case_history):
    y_true = []
    y_score = []
    for case in case_history:
        if case.get('ground_truth') is not None and case.get('binary_prediction') is not None:
            gt = np.array(case['ground_truth']).flatten()
            pred = np.array(case['binary_prediction']).flatten()
            y_true.extend(gt)
            y_score.extend(pred)
    if y_true and y_score:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_true, np.array(y_score) > 0.5)
        return {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc, 'cm': cm.tolist()}
    return None

@app.route('/analytics')
@login_required
def analytics():
    # Load analytics data
    if os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH, 'r') as f:
            analytics_data = json.load(f)
    else:
        analytics_data = {'total_images': 0, 'tumor_percentages': [], 'recent_cases': []}
    # Calculate stats
    tumor_percentages = analytics_data['tumor_percentages']
    stats = {}
    if tumor_percentages:
        stats = {
            'mean': np.mean(tumor_percentages),
            'median': np.median(tumor_percentages),
            'std': np.std(tumor_percentages),
            'min': np.min(tumor_percentages),
            'max': np.max(tumor_percentages)
        }
    # Advanced analytics: ROC, confusion matrix, time trends
    case_history = session.get('case_history', [])
    advanced_metrics = compute_roc_and_confusion(case_history)
    return render_template('analytics.html', analytics=analytics_data, stats=stats, advanced_metrics=advanced_metrics)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

if __name__ == '__main__':
    app.run(debug=True) 