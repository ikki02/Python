FROM python:3.7.6-stretch

RUN apt-get update && apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 swig
RUN pip install --upgrade pip
RUN pip install flask mecab-python3

WORKDIR /deploy
COPY . /deploy/

EXPOSE 9999

#ENTRYPOINT ["./boot.sh"]
