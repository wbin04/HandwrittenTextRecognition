import base64
import cv2
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from io import BytesIO
from PIL import Image
from TextRecognition import VietOCR

app = Flask(__name__)

def resize_to_fixed_canvas(image, target_size=(300, 150)):
    w, h = image.size
    scale = min(target_size[0] / w, target_size[1] / h)  
    new_w, new_h = int(w * scale), int(h * scale)

    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)  
    new_image = Image.new("RGB", target_size, (255, 255, 255))  
    new_image.paste(image, ((target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2))  

    return new_image

@app.route("/recognize", methods=["POST"])
def recognize_text():
    data = request.get_json()
    print("recognize")
    if "image" not in data:
        return jsonify({"error": "No image data received"}), 400
    
    try:
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        image.save("received_image.png")  
        # image = io.BytesIO(image_data)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
    
    image = resize_to_fixed_canvas(image)
    
    ocr = VietOCR(device='cpu')
    text = ocr.recognize_text(image)
    
    print("OCR Output:", repr(text))  
    print("Done")

    return jsonify({"text": text})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
