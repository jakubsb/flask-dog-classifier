FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN pip install torch==1.12.0+cpu torchvision==0.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
EXPOSE 8000
CMD ["python", "app.py"]
