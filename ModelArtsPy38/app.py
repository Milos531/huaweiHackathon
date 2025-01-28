from flask import Flask, request, jsonify
import json
from inference import inference
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/greet', methods=['POST'])
def say_hello_func():
    print("-------- in hello func ----------")
    data = json.loads(request.get_data(as_text = True))

    print(data)
    username = data['name']
    rsp_msg = f'Hello, {username}!'
    return jsonify(repsonse = rsp_msg), 200

@app.route('/goodbye', methods=['GET'])
def say_goodbye_func():
    print("------------ in goodbye func -------------")
    return '\nGoodbye!\n'


@app.route('/', methods=['POST'])
def default_func():
    print("----------- in default func -----------")
    data = json.loads(request.get_data(as_text=True))
    return f'\n called default func !\n {str(data)} \n'

@app.route('/stroke', methods=['POST'])
def predict_stroke():
    data = json.loads(request.get_data(as_text=True))
    res = inference(data)
    response = jsonify(result= res)
    # response = response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)