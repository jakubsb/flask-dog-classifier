import os
import sys
import io
from PIL import Image
import pytest

# to ensure module is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_image_file():
    img = Image.new("RGB", (100, 100), color="black")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_byte_arr.name = "test_image.png"
    
    return img_byte_arr