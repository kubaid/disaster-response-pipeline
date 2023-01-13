# Disaster Response Pipeline

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Motivation](#project-motivation)
4. [File description](#file-description)
5. [Licensing](#licensing)

## Installation

The code was written in Python 3.10. All the necessary libraries are listed in the requirements.txt file and can be installed using the following command:
`pip install -r requirements.txt`.

## Project Motivation

In this project I have implemented an ETL (Extract Transform Load) pipeline and trained a simple ML (Machine Learning) model to classify messages sent during disasters. The purpose of this project is to facilitate disaster response by sending categorized messages to proper emergency services.

This project was made as part of the Udacity's Data Science Nanodegree.

## Usage

First, run the ETL pipeline specifying datasets and resulting database name:
```python
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

Next, create and export the model by running:
```python
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

## File Description

The repository has the following structure:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- process_data.py # ETL pipeline

- models
|- train_classifier.py # classification model
```

## Licensing

This project is licensed under the [MIT license](LICENSE). Parts of the project (most notably the dashboard app) were provided by Udacity. 