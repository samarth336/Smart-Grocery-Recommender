from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model components from the single pickle file
with open("grocery_recommendation_model.pkl", "rb") as f:
    model_data = pickle.load(f)

item_similarity_df = model_data["item_similarity_df"]
tag_similarity_df = model_data["tag_similarity_df"]
seasonal_popularity = model_data["seasonal_popularity"]
user_item_matrix = model_data["user_item_matrix"]

# ---------------- Recommendation Function ----------------
def recommend_items(user_id, season, top_n=5):
    # Get items the user has already interacted with
    user_purchases = user_item_matrix.loc[user_id]
    purchased_items = user_purchases[user_purchases > 0].index.tolist()

    scores = {}

    for item in purchased_items:
        # Item similarity score
        similar_items = item_similarity_df[item].drop(labels=[item]).sort_values(ascending=False)
        for sim_item, score in similar_items.items():
            if sim_item not in purchased_items:
                scores[sim_item] = scores.get(sim_item, 0) + score * 0.5  # weight = 0.5

        # Tag similarity score
        if item in tag_similarity_df:
            similar_tags = tag_similarity_df[item].drop(labels=[item]).sort_values(ascending=False)
            for tag_item, score in similar_tags.items():
                if tag_item not in purchased_items:
                    scores[tag_item] = scores.get(tag_item, 0) + score * 0.3  # weight = 0.3

    # Seasonal boost
    seasonal_items = seasonal_popularity[(seasonal_popularity['season'] == season)]
    for _, row in seasonal_items.iterrows():
        item = row['item']
        if item not in purchased_items:
            scores[item] = scores.get(item, 0) + row['purchase_count'] * 0.01  # weight = 0.01

    # Sort scores and return top N
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in recommended[:top_n]]

# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    user_ids = user_item_matrix.index.tolist()
    recommendations = []

    if request.method == 'POST':
        user_id = request.form['user_id']
        season = request.form['season']
        recommendations = recommend_items(user_id, season)

    return render_template('index.html', user_ids=user_ids, recommendations=recommendations)

# ---------------- Run Server ----------------
if __name__ == '__main__':
    app.run(debug=True)
