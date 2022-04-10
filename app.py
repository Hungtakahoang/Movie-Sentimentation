from flask import Flask, request, render_template, url_for
from Inferences import inference_model
from Model import RNN
from DataExtract import dataExtract

app = Flask(__name__)

@app.route('/')
def reviewSubmit():
    return render_template('main_page.html')

@app.route('/result', methods=['POST'])
def reviewEvaluation():
    sentence = request.form['review']
    value_1, value_2 = inference_model(sentence)
    predict = [value_1, value_2]
    return render_template('result.html', data=predict)

if __name__ == "__main__":
    app.run(debug=True)
