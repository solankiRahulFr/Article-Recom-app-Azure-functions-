from flask import Flask, jsonify
import numpy as np
import pickle
import os
import lightfm
# Always use relative import for custom module
from .package.module import MODULE_VALUE
from azure.storage.blob import BlobClient

app = Flask(__name__)
rootPath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
connection_string = "DefaultEndpointsProtocol=https;AccountName=project09group8aa0;AccountKey=8P39lFEsZcCnUDQe8b0cg4izw4JCp1kiVETWh6/sm/WEFKgNYQsD3HALdtZSN+C640t8ufQKGLna+AStEbySIg==;EndpointSuffix=core.windows.net"


def read_pickle_from_azure(blob_name):
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="picklefiles", blob_name)
    print("Downloading..."+blob_name)
    with open("models/"+blob_name, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)


def loadModule():
    # global load_model, load_interactions, load_item_features_matrix, load_item_dict
    read_pickle_from_azure("lightfm_model_hybrid.pkl")
    read_pickle_from_azure("interactions.pkl")
    read_pickle_from_azure("item_features_matrix.pkl")
    read_pickle_from_azure("item_dict.pkl")
    # load_model = pickle.load(open(rootPath + '/FlaskApp/app_modules/lightfm_model_hybrid.pkl','rb'))
    # load_interactions = pickle.load(open(rootPath + '/FlaskApp/app_modules/interactions.pkl','rb'))
    # load_item_features_matrix = pickle.load(open(rootPath + '/FlaskApp/app_modules/item_features_matrix.pkl','rb'))
    # load_item_dict = pickle.load(open(rootPath + '/FlaskApp/app_modules/item_dict.pkl','rb'))



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
    check = os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/FlaskApp/app_modules/lightfm_model_hybrid.pkl')
    return jsonify(check=check)

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
