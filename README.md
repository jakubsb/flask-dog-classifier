# Dog Breed Classifier

This repository contains a two-container Docker application for dog breed prediction:

- `frontend`: Nginx serving static UI files
- `backend`: Flask REST API running with Gunicorn

## Project structure

- `backend/app.py` - Flask API routes (`/api/v1/predict`)
- `backend/model/image_classification.py` - image preprocessing and model inference
- `backend/model/model_dict.pth` - trained model weights
- `backend/labels/labels.txt` - class labels used by the model
- `backend/Dockerfile` - backend container build/runtime configuration
- `backend/requirements.txt` - backend Python dependencies
- `frontend/templates/index.html` - static frontend page
- `frontend/static/` - frontend CSS and JavaScript assets
- `frontend/nginx/default.conf` - Nginx static/proxy configuration
- `frontend/Dockerfile` - frontend container build configuration
- `docker-compose.yml` - service orchestration

## Requirements

- Docker Desktop (running)

## Run the application

1. Clone the repository.
2. Open a terminal in the project root.
3. Build and start both containers:

```bash
docker compose up --build
```

4. Open the app at [http://localhost:8000](http://localhost:8000).

## Stop containers

```bash
docker compose down
```


