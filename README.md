# Flask app for image classification

The following repository contains a containerized web application with Docker for identifying a dog breed from an input picture.

## Repository structure

- **model/image_classification.py**: contains functions for processing and labeling the input image
- **model/model_dict.pth**: serialized state dictionary of trained model
- **app.py**: structure for a flask app
- **Dockerfile**: allows to build an image containing code and necessary libraries
- **labels/labels.txt**: contains all labels used for prediciton
- **definitions.py** contains variable **ROOT_DIR** for root directory of this project
- **requirements.txt**: required libraries (other than pytorch libraries)
- **templates**, **static**: static folder contains assets used by templates to build a web app UI

## Getting started
1. Ensure you have Docker installed on your machine.
2. Clone this repository.
3. Navigate to the repository directory.
4. Run **docker build --tag=flask_dog_classification .**. 
5. Run **docker run -d -p 8000:8000 flask_dog_classification**.
6. Navigate to **localhost:800** on your web browser.


