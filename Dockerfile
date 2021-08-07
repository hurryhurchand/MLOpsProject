FROM python:3.8


WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY train.py /app
COPY BostonData.csv /app


EXPOSE 8000

CMD ["python","./train.py"]
