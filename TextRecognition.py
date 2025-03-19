from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import io

class VietOCR:
    def __init__(self, device='cpu'):
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['device'] = device
        self.detector = Predictor(self.config)
    
    def recognize_text(self, image):
        try:
            if isinstance(image, bytes):  
                image = Image.open(io.BytesIO(image))
            # elif isinstance(image, io.BytesIO):  
            #     image = Image.open(image)
            elif not isinstance(image, Image.Image):  
                return "Error: Invalid image format"
            
            text = self.detector.predict(image)
            return text
        except Exception as e:
            return f"Error: {str(e)}"
        
    def recognize_text_from_image_path(self, image_path):
        try:
            image = Image.open(image_path)
            text = self.detector.predict(image)
            return text
        except Exception as e:
            return f"Error: {str(e)}"