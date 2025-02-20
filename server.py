import sqlite3
import json
import os
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory,render_template
import cv2
import numpy as np
from subpixel_edges import subpixel_edges
import base64
np.bool = bool

app = Flask(__name__)
CORS(app)

CONFIG_FILE = 'config.json'
DB_FILE = 'measurement_tool.db'
UPLOAD_FOLDER = 'uploads'
OUTPUT_DIR = 'output_images'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS measurements (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            image_path TEXT NOT NULL,
                            json_data TEXT NOT NULL)''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}")

def save_to_config(new_entry):
    try:
        if os.path.exists(CONFIG_FILE) and os.path.getsize(CONFIG_FILE) > 0:
            with open(CONFIG_FILE, 'r') as file:
                data = json.load(file)
        else:
            data = []

        if not isinstance(data, list):
            data = []

        for i, item in enumerate(data):
            if item.get('partName') == new_entry.get('partName'):
                data[i] = new_entry
                break
        else:
            data.append(new_entry)

        with open(CONFIG_FILE, 'w') as file:
            json.dump(data, file, indent=4)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json")
    except Exception as e:
        print(f"Error saving to config: {e}")

@app.route('/save-config', methods=['POST'])
def save_config():
    try:
        print("In save-config route")

        print("Request Files:", request.files)
        print("Request Form:", request.form)

        json_data = request.form.get('json')
        if not json_data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

        try:
            new_entry = json.loads(json_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return jsonify({'success': False, 'error': 'Invalid JSON format'}), 400

        required_keys = ["partName", "parameters", "boundingBoxes"]
        if not all(key in new_entry for key in required_keys):
            return jsonify({'success': False, 'error': 'Missing required fields in JSON'}), 400

        save_to_config(new_entry)

        return jsonify({'success': True, 'message': 'Data saved to config.json'}), 200

    except Exception as e:
        print(f"Error saving config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-config', methods=['GET'])
def get_config():
    try:
        if os.path.exists(CONFIG_FILE) and os.path.getsize(CONFIG_FILE) > 0:
            with open(CONFIG_FILE, 'r') as file:
                return jsonify(json.load(file)), 200
        return jsonify([]), 200
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json")
        return jsonify({'error': 'Invalid JSON in config'}), 500
    except Exception as e:
        print(f"Error loading config: {e}")
        return jsonify({'error': 'Failed to load config'}), 500

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files or 'partName' not in request.form:
            return jsonify({"error": "Image and partName are required"}), 400

        image_file = request.files['image']
        part_name = request.form['partName']

        np_img = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        config = load_config()
        bboxes = get_bounding_boxes(config, part_name)

        if not bboxes:
            return jsonify({"error": f"Bounding box for part '{part_name}' not found"}), 404

        for box in bboxes:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = img_gray[y:y + h, x:x + w]

            edges = subpixel_edges(roi, 25, 0, 2)

            if hasattr(edges, 'x') and hasattr(edges, 'y'):
                for ex, ey in zip(edges.x, edges.y):
                    ex, ey = int(ex + x), int(ey + y)
                    cv2.circle(img, (ex, ey), 2, (0, 0, 255), -1)

        cv2.imwrite("img.png", img)
        _, img_encoded = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return jsonify({"message": "Image processed successfully", "processed_image": f"data:image/png;base64,{img_base64}"}), 200
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500





@app.route('/check_length', methods=['POST'])
def check_length():
    try:
        print("In calibration route")

        if 'image' not in request.files or 'json' not in request.form:
            return jsonify({"error": "Image and JSON payload are required"}), 400

        image_file = request.files['image']
        payload = json.loads(request.form['json'])

        if 'boundingBoxes' not in payload or 'parameters' not in payload:
            return jsonify({"error": "Bounding boxes and parameters are required in the JSON"}), 400

        roi_data = payload['boundingBoxes']
        parameters = payload['parameters']

        np_img = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE).astype(float)

        results = []

        for roi, param in zip(roi_data, parameters):
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

            if w <= 0 or h <= 0 or x < 0 or y < 0:
                param['cal'] = None
                continue

            cropped_roi = img[y:y + h, x:x + w]
            edges = subpixel_edges(cropped_roi, 25, 0, 2)

            if hasattr(edges, 'y') and len(edges.y) >= 2:
                vertical_diff = np.max(edges.y) - np.min(edges.y)

                cal_value = float(param.get('cal', 1))
                length = vertical_diff * cal_value if vertical_diff != 0 else None
                print("vertical_diff:",vertical_diff)
                print("cal_value:",cal_value)
                print("length:",length)
                results.append({"partName": param["name"], "length": length})

        return jsonify({"measured_lengths": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500





@app.route('/choose', methods=['GET', 'POST'])
def choose():
    return render_template('choose.html')
@app.route('/', methods=['GET', 'POST'])
def into():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/calibrate', methods=['POST'])
def calibrate():
    try:
        print("In calibration route")

        for key, file in request.files.items():
            print(f"File Key: {key}, Filename: {file.filename}")

        print("Form Data:")
        for key, value in request.form.items():
            print(f"{key}: {value}")

        if 'image' not in request.files or 'json' not in request.form:
            print("Not accepted ")
            return jsonify({"error": "Image and JSON payload are required"}), 400

        image_file = request.files['image']

        payload = json.loads(request.form['json'])

        if 'boundingBoxes' not in payload or 'parameters' not in payload:
            return jsonify({"error": "Bounding boxes and parameters are required in the JSON"}), 400

        roi_data = payload['boundingBoxes']
        parameters = payload['parameters']

        np_img = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE).astype(float)

        for roi, param in zip(roi_data, parameters):
            print("I am inside")
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']

            if w <= 0 or h <= 0 or x < 0 or y < 0:
                print(f"Skipping invalid ROI: {roi}")
                param['cal'] = None
                continue

            cropped_roi = img[y:y + h, x:x + w]

            edges = subpixel_edges(cropped_roi, 25, 0, 2)

            if hasattr(edges, 'y') and len(edges.y) >= 2:
                vertical_diff = np.max(edges.y) - np.min(edges.y)

                known_length = float(param.get('value', 1))
                print("Vertical diff:",vertical_diff)
                print("Known length:",known_length)
                calibration_ratio = known_length / vertical_diff if vertical_diff != 0 else None
                print("calibrated value:",calibration_ratio)
                param['cal'] = calibration_ratio
            else:
                param['cal'] = None

        payload['parameters'] = parameters
        save_to_config(payload)

        print("Updated Parameters with Calibration:", parameters)

        return jsonify({"parameters": parameters}), 200

    except Exception as e:
        print(f"Error in calibration: {e}")
        return jsonify({"error": str(e)}), 500


def load_config():
    try:
        if os.path.exists(CONFIG_FILE) and os.path.getsize(CONFIG_FILE) > 0:
            with open(CONFIG_FILE, 'r') as file:
                return json.load(file)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json")
    except Exception as e:
        print(f"Error loading config: {e}")
    return []

def get_bounding_boxes(config, part_name):
    for item in config:
        if item.get('partName') == part_name:
            return item.get('boundingBoxes', [])
    return []

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
