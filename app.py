from flask import Flask, render_template, request
from google.protobuf import message
from preprocessing import *


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST','GET'])

def main():
    if request.method == "POST":
        #query_title = request.form.get("title")
        #query_text = request.form.get("maintext")
        query = request.form.get("title")
        print("PROCESSED:"+query)
        preprocessed_input = preprocess(query)
        print("Afterrr:")
        print(preprocessed_input)
        
        gs1=joblib.load("model_gs.pkl")
        predd = gs1.predict(preprocessed_input)
        
        print(predd[0])
        if predd[0] == 1 :
            return render_template('index.html', message="False News")
        else:
            return render_template('index.html', message="True News")


if __name__ == '__main__':
    app.run(port=2080, debug=True)