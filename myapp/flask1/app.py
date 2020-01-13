import os
import subprocess
import traceback
import re
import MeCab
from flask import Flask, request, abort, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pattern = r"(名詞)|(?<!助)動詞|(形容詞)"
doc_morpheme_list = []

app = Flask(__name__, static_folder=".", static_url_path="")
app.config['JSON_AS_ASCII'] = False


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/parse", methods=["POST"])
def predict():
    print("parse input text.")

    try:
        m = MeCab.Tagger("-Ochasen")
        print("using default dict for text parser.")
    except Exception as e:
        print(e)

    if not request.is_json:
        abort(400, {"message": "Input Content-Type is not application/json."})

    data = request.get_json()
    if "text" not in data:
        abort(400, {"message": "text is not present in request parameter."})

    text = data["text"]
    if not isinstance(text, str):
        abort(400, {"message": "text is not string."})

    try:
        text_tokenized = m.parse(text).rstrip("EOS\n")
        doc_elements = [elements.split("\t") for elements in text_tokenized.splitlines()]
        for element in doc_elements:
            try:
                if re.search(pattern, element[3]):
                    doc_morpheme_list.append(element[2]) # 原形を抽出。表層形はelement[0]。品詞はelement[3]
            except:
                pass
        base_list = [" ".join(doc_morpheme_list)]
    except Exception as e:
        abort(500, {"message": "parsing error occurred: {}".format(e)})

    output_json = jsonify({
        "code": 200,
        "message": None,
        "result": base_list,
    })
    return output_json


@app.errorhandler(400)
def bad_request_handler(error):
    output_json = jsonify({
        "code": error.code,
        "message": error.description["message"],
    })
    return output_json, error.code


@app.errorhandler(404)
def not_found_handler(error):
    output_json = jsonify({
        "code": error.code,
        "message": "Requested resource is not found.",
    })
    return output_json, error.code


@app.errorhandler(Exception)
def internal_server_error_handler(e):
    print(traceback.format_exc())
    output_json = jsonify({
        "code": 500,
        "message": traceback.format_exc(),
    })
    return output_json, 500


if __name__ == "__main__":
    app.run() #hostは指定する必要がある。portは任意で構わないが、セキュリティグループの設定を忘れずに。
