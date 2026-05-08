FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
