# COMP49X-24-25-insect-detection-project

Convolutional Neural Network repository for training and testing seed beetle image classification models.

## Description

### Features 3 training programs

#### training_program.py: 
Our current implementation for the web app. Builds a seperate model for each 4 image angle. Has a parameter setting for training species or genus.

#### alt_training_program.py: 
Alternate models trained in the same way, but combining multiple image angles together (lateral and caudal, lateral and dorsal, all).

#### post_eval_stack_training.py: 
Stacking model training for the combined outputs of the current implementation models.

#### transformation_classes.py: 
Contains pre-processing transformation classes to be used in both training and evaluation.

### Data Conversion/Loading

#### training_data_converter.py: 
Data converter that filters and transfers image data provided by Dr. Morse into an sqlite3 database

#### training_database_reader.py: 
Data reader that reads sqlite3 database into a pandas dataframe.

#### stack_dataset_creator.py: 
Modifies default dataframe to be able to input into stacking model for training and evaluation.

#### model_loader.py: 
Loads currently saved models in the models repository.

#### user_input_database.py: 
Currently unused. Program to create database that pulls in user-submitted images in order to grow the main training database.

### Model Evaluation

#### evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle species.

#### genus_evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle genus.

### Simulators

#### simulator.py: 
End-to-end testing simulator that runs full database conversion and reading, training based on user-input choice, and evaluation.

#### eval_simulator.py: 
Testing evaluation of currently implemented models using user input of images in the local dataset directory.

#### alt_training_simulator: 
Runs alt_training_program.py and trains models to assess accuracies.

#### stack_simulator.py: 
Testing simulator for training and evaluation of stack model.

#### globals.py: 
Contains global variable names for simulator file name references.

## Getting Started

### Dependencies

* pandas
* torch
* torchvision
* scikit-learn
* dill
* pylint
* python 3.11

### Installing

Evaluation simulators will only run properly with image dataset downloaded.

### Executing program

#### simulator.py

* Run in VSCode
* Repeatedly enter number corresponding to which models you'd like to train
* Wait for training and automatic evaluation results

#### eval_simulator.py

* Run in VSCode
* Enter specimen id of specimen in local dataset that you'd like to evaluate e.g. GEM_187675032

#### stack_simulator.py and alt_training_simulator.py

* Run in VSCode
* Automatically runs training and testing without input


PORT INSPECTOR PORTION

# COMP-49X-24-25-port-inspector-project

# Photo Upload and History Application

This project is a web application built with Django that allows users to upload photos of seed beetles, receive results as to their likely species and genus, and view their upload history. It provides a simple interface for photo management, enabling users to track their uploads conveniently.

## Features
The application satisfies the following user stories:

- As a Port Inspector I want to upload a photo of the beetles and be told the likely species’ of the specimen so I can soundly determine if the shipment can be let through
- As a user I want to be able to look back at my previous identifications and the results I received to guide myself going forward, and to have a record in case clarification or data is needed.

## Technologies Used
- **Backend & Frontend Framework:** Django
- **Database:** Django’s default SQLite 
- **Email API:** MailerSend

## API setup
If planning to host the application on a server, using an email API for account management (email verification, etc.) is required. Our application uses MailerSend but similar API's can be used.  

### 1. Open account with MailerSend
Make an account with [MailerSend](https://www.mailersend.com/), verify the domain you   
would like to host the server on, and generate credentials for SMTP sending.

### 2. Put Credentials in Config File
With the MailerSend credentials, fill out the `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` variables

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/usd-cs/COMP-49X-24-25-port-inspector-project.git)
```

### 2. Navigate to the Project Directory
```bash
cd port_inspector
```

### 3. Install Dependencies
Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)  
With Docker Desktop open, run the following command in the project directory.
```bash
docker compose build
```

## Running the Server

### 1. Start the Server
Use docker to run migrations and start up the server by running the following:
```bash
docker compose up
```

### 2. Access the Application Locally
Open a web browser and go to the following link:
```
http://localhost:8000/
```


