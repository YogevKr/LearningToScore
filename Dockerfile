FROM pytorch/pytorch:latest

COPY . /LearningToScore

WORKDIR /LearningToScore

RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt
