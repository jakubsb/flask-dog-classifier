def test_health_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_success_with_valid_file(client, valid_image_file):
    response = client.post(
        "/api/v1/predict",
        data={"file": valid_image_file},
        content_type="multipart/form-data",
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert set(payload.keys()) == {"breed", "confidence"}


def test_predict_returns_400_when_file_is_missing(client):
    response = client.post("/api/v1/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "No file uploaded"}


def test_predict_returns_500_on_inference_failure(client, monkeypatch, valid_image_file):
    import app as app_module

    def raise_inference_error(_input_tensor):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(app_module, "predict_class", raise_inference_error)

    response = client.post(
        "/api/v1/predict",
        data={"file": valid_image_file},
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    assert response.get_json() == {"error": "inference failed"}


def test_predict_returns_400_for_unsupported_file_type(client, invalid_file_type):
    from app import UNSUPPORTED_FILE_TYPE_ERROR

    response = client.post(
        "/api/v1/predict",
        data={"file": invalid_file_type},
        content_type="multipart/form-data",
    )

    payload = response.get_json()

    assert response.status_code == 400
    assert payload == {"error": UNSUPPORTED_FILE_TYPE_ERROR}
    assert "breed" not in payload
    assert "confidence" not in payload
