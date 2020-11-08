# ML_Disaster_Response_Pipeline
A Machine Learning Pipeline to categorize emergency messages based on the needs communicated by the sender. Project includes Flask Web Application.

### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions: <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


The project was created with Python 3.8.0.
Install the virtual environment with the rquired python libraries using
requirements.txt.
xxxx

## Project Motivation: <a name="motivation"></a>

Motivation of this project is to implement a machine learning pipeline
including Natural Language Processing using the Python Scikit-Learn library.
This corporate message data is from one of the free datasets provided on the
[Figure Eight Platform](https://appen.com/resources/datasets/), licensed under
a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
.


## File Descriptions: <a name="files"></a>
xxx

```
ML_Disaster_Response_Pipeline/
│
├── README.md
├── app/
│   ├── template/
│   │   ├── master.html  # main page of web app
│   │   ├── go.html  # classification result page of web app
│   ├── run.py  # Flask file that runs app
│
├── data/
│   ├── disaster_categories.csv  # data to process
│   ├── disaster_messages.csv  # data to process
│   ├── process_data.py
│   ├── InsertDatabaseName.db   # database to save clean data to
│
├── models/
│   ├── train_classifier.py
│   ├── classifier.pkl  # saved model

```


## Results: <a name="results"></a>
The results can be seen locally with following the steps:
* create virtual environment in folder **ML_Disaster_Response_Pipeline/**:
  - `python3 -m venv disaster_response_env`
* activate the virtual environment:
  - `source disaster_response_env/bin/activate`
* pip install required packages:
  - `pip install -r requirements.txt`
* change directory to **ML_Disaster_Response_Pipeline**:
  - `cd ML_Disaster_Response_Pipeline'
* start Flask Web App:
  - python xxx.py
* Now open in your browser the following url:
  - `http://0.0.0.0:3001/`

## Licensing, Authors, Acknowledgements: <a name="licensing"></a>

This corporate message data is from one of the free datasets provided on the
[Figure Eight Platform](https://appen.com/resources/datasets/), licensed under
a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Feel free to use my code as you please:

Copyright 2020 Leopold Walther

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
