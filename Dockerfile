FROM pytorch/pytorch:latest

COPY ./requirements.txt /LearningToScore/requirements.txt
COPY ./requirements-dev.txt /LearningToScore/requirements-dev.txt

WORKDIR /LearningToScore

RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

COPY ./learning_to_score /LearningToScore/learning_to_score
COPY ./research /LearningToScore/research

ENTRYPOINT ["sh", "./research/sweep.sh" ]