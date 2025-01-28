FROM python:3

RUN mkdir -p /opt/src/app/model
WORKDIR /opt/src/app

COPY  inference.py ./inference.py
COPY app.py ./app.py
COPY requirements.txt ./requirements.txt
COPY model/model_weights.pt ./model/model_weights.pt

RUN pip install -r ./requirements.txt
ENV PYTHONPATH="/opt/src/admin"

ENTRYPOINT ["python", "./app.py"]
