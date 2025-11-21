# Seed Beetle Identification App

Convolutional Neural Network repository for training and testing seed beetle image classification models.
Also a django app development repository for creating and testing new features for easy public usage of the models 
developed by the CNN side of the project.

## Description

### Features 4 training programs

#### training_program.py: 
Our current implementation for the web app. Builds a seperate model for each 4 image angle. Has a parameter setting for training species or genus. Utilizes 
Resnet50 models.

#### alt_training_program.py: 
Alternate models trained in the same way, but combining multiple image angles together (lateral and caudal, lateral and dorsal, all).

#### post_eval_stack_training.py: 
Stacking model training for the combined outputs of the current implementation models using a linear regression model.

#### yolo_training_program.py:
Object detection model program that trains a YoloV8 model to recognize a singular object (beetles in this case) to allow easy cropping of images to the main subject.

#### transformation_classes.py: 
Contains pre-processing transformation classes to be used in both training and evaluation.

### Data Conversion/Loading

#### training_data_converter.py: 
Data converter that filters and transfers image data provided by Dr. Morse into an sqlite3 database. Expects data in the form of "Genus species UUID ext view.jpg". 
Any non-matching files are dropped and listed in the terminal for ease of access when finding misnamed files. Depending on the format of image data and parameters being 
used, the table and the expected fields should be changed to properly reflect the data's format.

#### training_database_reader.py: 
Data reader that reads sqlite3 database into a pandas dataframe. This database is structured to match the converter's format, so changes to the table for the converter must 
also be made on the reader to ensure proper conversion. The current dataset this program is set to use follow a file naming scheme of: "Genus species SpecimenUUID resolution angle.jpg"

#### stack_dataset_creator.py: 
Modifies default dataframe to be able to input into stacking model for training and evaluation. This dataset creator relies on certain parts of the image name that was input, so 
modifications must be made to match the data input. The current setting is for the same format as the scheme above.

#### model_loader.py: 
Loads currently saved models in the models repository for evaluation of images.

#### user_input_database.py: 
Currently not used. Program to create database that pulls in user-submitted images in order to grow the main training database. The user's entries are handled by the 
app itself, so this database is no longer in use in the project, but is available as a step in between user input and proper adoption into the training database.

### Model Evaluation

#### evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle species. There are multiple options for which form of evaluation to use 
which can be decided upon initialization of the evaluation method class. The method will return the top 5 species predictions.

#### genus_evaluation_method.py: 
Uses necessary models based off input angles provided to evaluate and classify beetle genus. The same options exist as the species evaluation method above, but 
the genus method only returns the best fit option.

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
* use requirements.txt to install the latest list of dependencies reliably

### Installing

Evaluation simulators will only run properly with image dataset downloaded. Be sure that the dataset is stored in the base repository directory and contains only images 
with a proper naming scheme to match what the data converter expects.

### Executing program

#### simulator.py

* Run in VSCode
* Enter numbers corresponding to which models you'd like to train and the other options available
* Wait for training and automatic evaluation results

#### eval_simulator.py

* Run in VSCode
* Enter specimen id of specimen in local dataset that you'd like to evaluate e.g. GEM_187675032

#### stack_simulator.py and alt_training_simulator.py

* Run in VSCode
* Automatically runs training and testing without input


# App development portion of repository

# Photo Upload and History Application

This project is a web application built with Django that allows users to upload photos of seed beetles, receive results as to their likely species and genus, and view their upload history. It provides a simple interface for photo management, enabling users to track their uploads conveniently. It also features an admin page edited from the django default 
to allow management of the app.

## Features
The application includes:
- A homepage and about page for information on the site
- Login and email verification to ensure users are accountable and to assign special statuses to port inspectors, the main target audience of the product
- A page that allows evaluation of images and a history page to track previous evaluations
- Admin page that allows the admin to verify the results of user's uploaded images, modify user's accounts except for email/password, and a features that allow the admin to 
download verified user inputs to add to the next training batch for the models
- A password reset feature in case a user forgets their password

## Technologies Used
- **Backend & Frontend Framework:** Django
- **Database:** Django’s default SQLite. For production, PostgreSQL is recommended instead and requires small edits in settings to implement
- **Email API:** Currently set to use basic SMTP, but settings allow configuration of an API like MailerSend instead

## API setup
If planning to host the application on a server, using an email API for account management (email verification, etc.) is required. Our application uses SMTP but APIs can be used. 
While developing locally, set the email to print in terminal instead of actual email sending for convenient testing. 

### 1. Open account with MailerSend
Make an account with [MailerSend](https://www.mailersend.com/), verify the domain you   
would like to host the server on, and generate credentials for SMTP sending.
If staying with the default SMTP, use the correct credentials in existing settings to enable use of your custom email account.
Visit this page for help on django's email functionality and setup: https://docs.djangoproject.com/en/5.2/topics/email/

### 2. Put Credentials in Config File
With the MailerSend credentials, fill out the `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` variables.
If using SMTP, use credentials given by the email provider.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BugAICorp/seed-beetle-identification-app.git
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
Alternatively, pip install everything in requirements.txt to test the app without using docker.

## Running the Server

### 1. Start the Server
Use docker to run migrations and start up the server by running the following:
```bash
docker compose up
```
If not using docker, run py manage.py makemigrations, py manage.py migrate, and then py manage.py runserver to 
test the server locally without using docker. Ensure localhost is an allowed host in settings for development testing.

### 2. Access the Application Locally
Open a web browser and go to the following link:
```
http://localhost:8000/
```
If you are not using docker, a link will appear in the terminal to click instead.

## Current recommended server setup
The server's current implementation requires a backend and a web app with a celery worker communicating between the 
two. A Redis setup to communicate between the two ends is the recommended solution we are currently using. The backend 
hosts the AI models to allow the website to properly function while image evaluations happen in the background on separate 
hardware.



