from flask import Blueprint, request, jsonify
from app.recommender import recommend

recommender_routes = Blueprint("recommender_routes", __name__)

@recommender_routes.route("/recommend", methods=["GET"])
def get_recommendations():
    title = request.args.get("title", "")
    results = recommend(title)
    return jsonify(results)
