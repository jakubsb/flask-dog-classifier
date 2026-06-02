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


@pytest.fixture(params=["PNG", "JPEG", "WEBP"])
def valid_image_file(request):
    img = Image.new("RGB", (100, 100), color="black")
    buf = io.BytesIO()
    img.save(buf, format=request.param)
    buf.seek(0)
    buf.name = f"test_image.{request.param.lower()}"
    return buf


INVALID_FILE_CASES = [
    ("document.pdf", b"%PDF-1.4\n%%EOF\n"),
    ("program.exe", b"MZ\x90\x00"),
    ("animation.gif", b"GIF89a"),
    ("upload", b"binary-content"),
]


@pytest.fixture(params=INVALID_FILE_CASES, ids=["pdf", "exe", "gif", "no_extension"])
def invalid_file_type(request):
    filename, content = request.param
    buf = io.BytesIO(content)
    buf.name = filename
    return buf
