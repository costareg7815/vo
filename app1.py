import os
import requests
import configparser
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from moviepy.editor import ImageClip, AudioFileClip
from werkzeug.utils import secure_filename

# تحميل إعدادات الملف
config = configparser.ConfigParser()
config.read('config.ini')

UPLOAD_FOLDER = config['general']['UPLOAD_FOLDER']
VIDEO_FOLDER = config['general']['VIDEO_FOLDER']
ALLOWED_EXTENSIONS = config['general']['ALLOWED_EXTENSIONS'].split(', ')

XI_API_KEY = config['api']['XI_API_KEY']
BASE_API_URL = config['api']['BASE_API_URL']

female_1 = config['voices']['female_1']
female_2 = config['voices']['female_2']
male_1 = config['voices']['male_1']
female_3 = config['voices']['female_3']
male_2 = config['voices']['male_2']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = os.path.join(app.root_path, 'static', 'videos')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(app.config['VIDEO_FOLDER']):
    os.makedirs(app.config['VIDEO_FOLDER'])

VOICES = {
    'female_1': female_1,
    'female_2': female_2,
    'male_1': male_1,
    'female_3': female_3,
    'male_2': male_2
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_text_to_speech(text, voice_choice):
    try:
        if voice_choice not in VOICES:
            return None

        voice_id = VOICES[voice_choice]
        api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": XI_API_KEY
        }

        data = {
            "text": text,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.85
            }
        }

        response = requests.post(api_url, json=data, headers=headers)

        if response.status_code == 200:
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"output_{current_time}.mp3"
            file_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        else:
            print("Error:", response.status_code, response.text)
            return None
    except requests.exceptions.RequestException as e:
        print("API Error:", e)
        return None

def apply_effect_to_image(image_path, effect):
    image = cv2.imread(image_path)

    if effect == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif effect == 'blur':
        image = cv2.GaussianBlur(image, (15, 15), 0)
    elif effect == 'invert':
        image = cv2.bitwise_not(image)
    elif effect == 'edge':
        image = cv2.Canny(image, 100, 200)
    elif effect == 'resize':
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    elif effect == 'crop':
        h, w = image.shape[:2]
        image = image[h//4:3*h//4, w//4:3*w//4]
    elif effect == 'contrast':
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    elif effect == 'rotate':
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    elif effect == 'flip':
        image = cv2.flip(image, 1)
    elif effect == 'brightness':
        image = cv2.convertScaleAbs(image, alpha=1, beta=50)
    elif effect == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        image = cv2.transform(image, kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif effect == 'noise':
        noise = np.random.randint(0, 50, image.shape, dtype='uint8')
        image = cv2.add(image, noise)
    elif effect == 'motion_blur':
        size = 15
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)
    elif effect == 'gaussian_noise':
        row, col, ch = image.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
        image = cv2.add(image, gauss.astype('uint8'))
    elif effect == 'pencil_sketch':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray_image
        blur_img = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = 255 - blur_img
        sketch = cv2.divide(gray_image, inv_blur, scale=256.0)
        image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif effect == 'vignette':
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        for i in range(3):
            image[:, :, i] = image[:, :, i] * mask
    # يمكنك إضافة باقي التأثيرات هنا

    # حفظ الصورة المؤثرة مؤقتاً
    effected_path = image_path.replace(".", "_effected.")
    cv2.imwrite(effected_path, image)
    return effected_path

def create_video_with_audio(image_path, audio_file, output_filename):
    try:
        audio = AudioFileClip(audio_file)
        img_clip = ImageClip(image_path).set_duration(audio.duration)
        video = img_clip.set_audio(audio)
        video.write_videofile(output_filename, fps=24, codec='libx264')
        return output_filename
    except Exception as e:
        app.logger.error("Video creation error: %s", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_videos', methods=['POST'])
def create_videos():
    image_file = request.files['image']
    text = request.form['text']
    video_count = int(request.form['videoCount'])
    voice_choice = request.form['voice']
    effect = request.form.get('effect')

    if voice_choice not in VOICES:
        return jsonify({'error': 'Invalid voice choice'})

    if image_file and allowed_file(image_file.filename):
        image_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(image_filename)

        # تطبيق التأثير إذا تم اختياره
        if effect:
            image_filename = apply_effect_to_image(image_filename, effect)

        # تحويل النص إلى صوت
        audio_file = convert_text_to_speech(text, voice_choice)

        if audio_file:
            videos = []
            for i in range(video_count):
                video_filename = f'video_{i + 1}.mp4'
                output_filename = os.path.join(app.config['VIDEO_FOLDER'], video_filename)
                create_video_with_audio(image_filename, audio_file, output_filename)
                videos.append({'url': f'/static/videos/{video_filename}', 'name': video_filename})

            return render_template('Videos.html', videos=videos)
        else:
            return jsonify({'error': 'Text-to-speech failed'})
    else:
        return jsonify({'error': 'Invalid image file'})

@app.route('/get_existing_videos')
def get_existing_videos():
    videos = []
    for filename in os.listdir(app.config['VIDEO_FOLDER']):
        if filename.endswith('.mp4'):
            videos.append({'url': f'/static/videos/{filename}', 'name': filename})
    return render_template('voise.html', videos=videos)

@app.route('/static/videos/<filename>')
def send_video(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'videos'), filename)

if __name__ == '__main__':
    app.run(debug=True)
