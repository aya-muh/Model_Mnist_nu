import requests
import base64
import io
from PIL import Image, ImageDraw

def test():
    print("=== Testing Updated MNIST Verification Service ===")
    
    # 1. Health check
    r_health = requests.get('http://127.0.0.1:5000/health')
    print("Health Check:", r_health.status_code, r_health.json())

    # 2. Canvas test: Simulated drawing of digit 8
    pad = Image.new('RGB', (280, 280), (255, 255, 255))
    d = ImageDraw.Draw(pad)
    d.ellipse([(90, 40), (190, 140)], outline=(15, 23, 42), width=18)
    d.ellipse([(80, 130), (200, 250)], outline=(15, 23, 42), width=18)
    buf = io.BytesIO()
    pad.save(buf, format='PNG')
    b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

    r_canvas = requests.post('http://127.0.0.1:5000/predict', json={'canvas_data': b64})
    print("\n--- Canvas Input Result ---")
    print("Status:", r_canvas.status_code)
    data_c = r_canvas.json()
    print("Predicted Digit:", data_c.get('predicted_digit'))
    print("Confidence:", data_c.get('confidence'), "%")
    print("Inverted:", data_c.get('inverted'))
    print("Source:", data_c.get('input_source'))
    print("Latency:", data_c.get('inference_time_ms'), "ms")

    # 3. File upload test: test_4.png (digit 3)
    test_img_path = r'C:\Users\USER\OneDrive\Desktop\Aya\Machan learning\Machine_learning_Fils\test_4.png'
    with open(test_img_path, 'rb') as f:
        r_file = requests.post('http://127.0.0.1:5000/predict', files={'file': f})
    print("\n--- File Upload Result ---")
    print("Status:", r_file.status_code)
    data_f = r_file.json()
    print("Predicted Digit:", data_f.get('predicted_digit'))
    print("Confidence:", data_f.get('confidence'), "%")
    print("Inverted:", data_f.get('inverted'))
    print("Source:", data_f.get('input_source'))
    print("Latency:", data_f.get('inference_time_ms'), "ms")

if __name__ == '__main__':
    test()
