from flask import Flask, jsonify
import numpy as np
import _pickle as cPickle
import os
# Always use relative import for custom module
from .package.module import MODULE_VALUE

app = Flask(__name__)

def loadModules():
    print(os.path.dirname(os.path.realpath(__file__)), "-------------------------------------")
    global model, interactions, item_features_matrix, item_dict
    model = cPickle.load(open('./app_modules/lightfm_model_hybrid.pkl','rb'))
    interactions = cPickle.load(open('./app_modules/interactions.pkl','rb'))
    item_features_matrix = cPickle.load(open('./app_modules/item_features_matrix.pkl','rb'))
    item_dict = cPickle.load(open('./app_modules/item_features_matrix.pkl','rb'))



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


@app.route("/predictArticles/UserId/<id>", methods=['GET'])
def hello(id: int):
    loadModules()
    recom_articles, recom_categories=n_recommendation(model, interactions, id, item_dict, item_features_matrix)
    return jsonify(articles=recom_articles,
                   categories=recom_categories)

@app.route("/module")
def module():
    return f"loaded from FlaskApp.package.module = {MODULE_VALUE}"

if __name__ == "__main__":
    app.run()