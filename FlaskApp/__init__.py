from flask import Flask, jsonify
import numpy as np
import joblib
import joblib
import os
import lightfm
# Always use relative import for custom module
from .package.module import MODULE_VALUE

app = Flask(__name__)
rootPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def loadModule():
    global load_model, load_interactions, load_item_features_matrix, load_item_dict
    load_model = joblib.load(open(rootPath + '/FlaskApp/app_modules/lightfm_model_hybrid.pkl','rb'))
    load_interactions = joblib.load(open(rootPath + '/FlaskApp/app_modules/interactions.pkl','rb'))
    load_item_features_matrix = joblib.load(open(rootPath + '/FlaskApp/app_modules/item_features_matrix.pkl','rb'))
    load_item_dict = joblib.load(open(rootPath + '/FlaskApp/app_modules/item_dict.pkl','rb'))



def n_recommendation(model, interactions, user_id, item_dict , item_features_matrix, n_recommendations=5):
    n_users, n_items = interactions.shape
    # Get user and item mapping indices
    user_mapping = interactions.row
    item_mapping = interactions.col

    known_positives = item_mapping[user_mapping == user_id]
    # print(n_items)
    scores = model.predict(user_id, np.arange(n_items), item_features=item_features_matrix)

    # Sort the items in decreasing order of recommendation scores
    top_items = np.argsort(-scores)

    # Filter out the items that the user has already interacted with
    top_items = [item for item in top_items if item not in known_positives]

    articles = top_items[:n_recommendations]

    categories = [item_dict[article] for article in articles]

    return articles, categories

@app.route("/")
def index():
    return "main route working -- "+ os.listdir(rootPath)

@app.route("/predictArticles/<id>", methods=['GET'])
def predictArticles(id: int):
    loadModule()
    userid = int(id)
    recom_articles, recom_categories=n_recommendation(load_model, load_interactions, userid, load_item_dict, load_item_features_matrix, 5)
    response = jsonify(userid = userid,
                   articles=','.join(str(v) for v in recom_articles),
                   categories=','.join(str(v) for v in set(recom_categories)))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/module")
def module():
    return f"loaded from FlaskApp.package.module = {MODULE_VALUE}"

if __name__ == "__main__":
    app.run()
