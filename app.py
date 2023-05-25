from flask import Flask, jsonify, make_response, request, abort
import pandas as pd
import catboost 
import pickle
from flask_cors import CORS,cross_origin

model = pickle.load(open('E:\Pheonix\programing\Visual_studio_codes\Pythonpg\Machine_Learning\SalesPrediction\salesprediction_model.pkl','rb'))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error':'Not Found'}), 404)
@app.route("/")
def hello():
    return "Hello World!"

@app.route("/salesprediction/", methods=['POST'])
@cross_origin
def get_prediction():
    if not request.json: # Check whether request.json have a payload
        abort(400)  # aborts request with http status 400 (Bad request)
    df = pd.DataFrame(request.json, index=[0]) # index values says it should only have a single row
    cols=["CONSOLE","RATING","CRITICS_POINTS","CATEGORY","YEAR","PUBLISHER","USER_POINTS"]
    df = df[cols]
    return jsonify({'result' : model.predict(df)[0]}),201
                # [0] is used to access the first element of predicted values.
                # 201 is used to specify http status code (created)
if __name__ == "__main__":
    app.run()