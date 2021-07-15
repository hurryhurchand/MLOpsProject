FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip


WORKDIR /app

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY train.py /app
COPY BostonData.csv /app


EXPOSE 8000

CMD ["python3","-u","./train.py"]
