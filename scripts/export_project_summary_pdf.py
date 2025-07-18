from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Content for the PDF
lines = [
    'Tumor Detection Project Report',
    '',
    'Conceptual Overview',
    '',
    'Purpose:',
    'This project is designed to assist doctors, clinicians, and medical researchers in detecting and analyzing tumors from medical images (such as MRI, CT, or pathology slides) using deep learning.',
    '',
    'Problem Solved:',
    'Manual tumor detection is time-consuming and subject to human error. This system automates the process, providing fast, consistent, and accurate tumor segmentation and analytics.',
    '',
    'Target Users:',
    '- Doctors',
    '- Medical researchers',
    '- Clinicians',
    '',
    'High-Level Workflow:',
    '1. User uploads medical images.',
    '2. Images are preprocessed (cleaned, normalized, resized).',
    '3. A deep learning model (U-Net) performs tumor detection.',
    '4. Results are generated and visualized.',
    '5. Analytics dashboard summarizes results and trends.',
    '6. Users can download detailed PDF reports.',
    '',
    'Technical Overview',
    '',
    'Project Structure:',
    '- data/: Raw and processed images, masks',
    '- models/: Saved model weights and architectures',
    '- src/: Source code (data processing, models, training, inference, visualization)',
    '- web_app/: Web interface',
    '- config/: Configuration files',
    '- notebooks/: Jupyter notebooks for exploration',
    '- utils/: Utility functions',
    '',
    'Data Flow:',
    '- Raw Data: Uploaded or placed in data/raw/',
    '- Preprocessing: Cleaning, normalization, resizing, augmentation',
    '- Model Inference: U-Net model segments tumor regions',
    '- Results: Tumor percentage, area, overlays, and analytics are generated',
    '- Analytics: Dashboard shows statistics, distributions, and recent cases',
    '- Reporting: PDF reports can be generated and downloaded',
    '',
    'Key Technologies:',
    '- Python, Flask (web app)',
    '- TensorFlow/Keras (deep learning)',
    '- OpenCV, Albumentations (image processing & augmentation)',
    '- ReportLab (PDF generation)',
    '- Numpy, Scikit-learn (data analysis)',
    '',
    'Model Architecture:',
    '- U-Net: Encoder-decoder architecture with skip connections, ideal for medical image segmentation',
    '- Backbone: ResNet50 (configurable)',
    '- Loss: Dice Loss + Binary Crossentropy',
    '- Optimizer: Adam',
    '',
    'Data Preprocessing & Cleaning:',
    '- Normalization: Min-max, z-score, or histogram equalization',
    '- Resizing: To standard input size (configurable)',
    '- Augmentation: Rotation, flipping, brightness/contrast, etc.',
    '- Mask Binarization: Ensures masks are suitable for segmentation',
    '',
    'Training & Inference Pipeline:',
    '- Data split into train/validation/test',
    '- Model trained on preprocessed data',
    '- Inference pipeline preprocesses new images and predicts tumor regions',
    '',
    'Analytics & Reporting:',
    '- Dashboard: Shows total images, mean/median tumor %, distribution, recent cases',
    '- PDF Reports: Downloadable, include results and overlays',
    '',
    'Workflow Diagram:',
    '[See README or documentation for a visual diagram]',
]

def create_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont('Helvetica-Bold', 20)
    c.drawString(50, height - 50, lines[0])
    c.setFont('Helvetica', 12)
    y = height - 80
    for line in lines[1:]:
        if y < 60:
            c.showPage()
            c.setFont('Helvetica', 12)
            y = height - 50
        c.drawString(50, y, line)
        y -= 16
    c.save()

if __name__ == '__main__':
    create_pdf('project_summary_final.pdf') 