# 🦠 COVID19 Fake news detector 🥼🧬 [![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
The proliferation of fake news is a major challenge for modern democratic societies. Inaccurate information can affect people's health and well-being, especially during the difficult times of the COVID-19 pandemic. Furthermore, disinformation erodes public trust in democratic institutions by preventing citizens from making rational decisions based on verifiable facts.Here I have created this streamlit powered webapp which detects COVID19 related fake news.

### Live web-app can be found [here.](https://share.streamlit.io/anuvabsen1/covid_fake_news/app.py)


## Datasets:
* [COVID Fake News Dataset (Zenodo)](https://zenodo.org/record/4282522)
* [COVID19 Fake News Dataset (Mendeley Data)](https://data.mendeley.com/datasets/zwfdmp5syg/1)

## Installation:
Run ***pip install -r requirements.txt*** to install the dependencies.


## Usage:
1. Run the command: ***python train.py*** to train your spacy model.
2. Run the command: ***streamlit run app.py*** to run the webapp. This will launch your webapp in a new tab :



### Running the Dockerized App
1. Ensure you have Docker Installed and Setup in your OS (Windows/Mac/Linux). For detailed Instructions, please refer [this.](https://docs.docker.com/engine/install/)
2. Navigate to the folder where you have cloned this repository ( where the ***Dockerfile*** is present ).
3. Build the Docker Image (don't forget the dot!! :smile: ): 
```
docker build --tag fake_news_app .
```
4. Run the docker:
```
docker run --publish 8000:8080 --detach --name fapp fake_news_app
```

This will launch the dockerized app. Navigate to ***localhost:8000*** in your browser to have a look at your application. You can check the status of your all available running dockers by:
```
docker ps
```
