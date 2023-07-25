from flask import Flask, jsonify
import numpy as np
import _pickle as cPickle
import os
import lightfm
# Always use relative import for custom module
from .package.module import MODULE_VALUE

app = Flask(__name__)
rootPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def loadModules():
    load_model = cPickle.load(open(rootPath + '/FlaskApp/app_modules/lightfm_model_hybrid.pkl','rb'))
    load_interactions = cPickle.load(open(rootPath + '/FlaskApp/app_modules/interactions.pkl','rb'))
    load_item_features_matrix = cPickle.load(open(rootPath + '/FlaskApp/app_modules/item_features_matrix.pkl','rb'))
    load_item_dict = cPickle.load(open(rootPath + '/FlaskApp/app_modules/item_dict.pkl','rb'))
    return load_model, load_interactions, load_item_features_matrix, load_item_dict



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


@app.route("/predictArticles/<id>", methods=['GET'])
def predictArticles(id: int):
    userid = int(id)
    model, interactions, item_features_matrix, item_dict = loadModules()
    recom_articles, recom_categories=n_recommendation(model, interactions, userid, item_dict, item_features_matrix, 5)
    return jsonify(userid = userid,
                   articles=','.join(str(v) for v in recom_articles),
                   categories=','.join(str(v) for v in set(recom_categories)))

@app.route("/module")
def module():
    return f"loaded from FlaskApp.package.module = {MODULE_VALUE}"

if __name__ == "__main__":
    app.run()
