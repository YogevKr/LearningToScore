FROM pytorch/pytorch:latest

COPY ./requirements.txt /LearningToScore/requirements.txt
# COPY ./requirements-dev.txt /LearningToScore/requirements-dev.txt

WORKDIR /LearningToScore

RUN pip install -r requirements.txt
# RUN pip install -r requirements-dev.txt

COPY ./learning_to_score /LearningToScore/learning_to_score
COPY ./research /LearningToScore/research
COPY ./example.py example.py

COPY ./parkinsons_updrs.data /LearningToScore/ParkinsonVoiceDatasetLeaky/raw/parkinson_updrs.csv

CMD ["python", "-m", "example"]