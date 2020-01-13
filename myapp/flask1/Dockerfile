FROM python:3.7.6-stretch

RUN apt-get update && apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 swig
RUN pip install --upgrade pip && pip install flask mecab-python3

WORKDIR /deploy
COPY . /deploy/

ENV FLASK_APP "/deploy/app.py"

EXPOSE 9999

ENTRYPOINT ["flask"]
CMD ["run", "--host=0.0.0.0", "--port=9999"]
#サーバー上でdocker run -d -p 9999:9999 myflask
#ローカルでcurl http://eip:9999ってすればサーバ内コンテナのflaskサーバーにリクエストできる。
