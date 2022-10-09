FROM python:3.8

RUN set -ex && mkdir /translater
WORKDIR /translater

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model/ ./model
COPY . ./

EXPOSE 5000
ENV PYTHONPATH /translater
CMD python3 /translater/main.py