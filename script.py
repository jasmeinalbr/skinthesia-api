import pandas as pd
import ast
import logging
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# ------------------- SETUP LOGGING ------------------- #
def setup_logging():
    log_dir = "../logs"
    log_file = "recommendation_model.log"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging started for skincare recommendation system")

# ------------------- CATEGORY TO INGREDIENTS MAPPING ------------------- #
CATEGORY_TO_INGREDIENTS = {
    'brightening': ['niacinamide', 'vitamin c', 'arbutin', 'licorice'],
    'acne': ['salicylic acid', 'tea tree'],
    'hydrating': ['hyaluronic acid', 'glycerin'],
    'calming': ['centella asiatica', 'green tea', 'aloe vera', 'ceramide', 'panthenol'],
    'anti_aging': ['vitamin e', 'zinc', 'retinol'],
    'exfoliant': ['aha', 'bha', 'pha', 'lactic acid', 'glycolic acid', 'mandelic acid']
}

def map_category_to_ingredients(categories):
    """Mengubah daftar kategori menjadi daftar bahan spesifik."""
    ingredients = set()
    for category in categories:
        if category in CATEGORY_TO_INGREDIENTS:
            ingredients.update(CATEGORY_TO_INGREDIENTS[category])
        else:
            logging.warning(f"Category '{category}' not found in mapping. Using as-is.")
            ingredients.add(category)
    return list(ingredients)

# ------------------- CLASSIFICATION MODEL INFERENCE ------------------- #
def clean_input(lst):
    if isinstance(lst, str):
        lst = [lst]
    seen = set()
    result = []
    for item in lst:
        item_norm = re.sub(r"[\\[\\]\'\"]", "", item.lower().strip())
        if item_norm and item_norm not in seen:
            seen.add(item_norm)
            result.append(item_norm)
    return result

# Updated class weights
class_weights_np = np.array([0.98684211, 1.0, 0.87209302, 1.0, 1.0, 1.0])
class_weights_tf = tf.constant(class_weights_np, dtype=tf.float32)

def weighted_binary_crossentropy(y_true, y_pred):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * class_weights_tf + (1 - y_true) * 1.0
    weighted_bce = bce * weight_vector
    return tf.keras.backend.mean(weighted_bce)

def predict_ingredient_categories(
    skin_type,
    skin_concern,
    skin_goal,
    ingredient,
    threshold=0.3,
    use_class_thresholds=False,
    best_thresholds=None
):
    try:
        with open("src/models/mlb_classes.json", "r") as f:
            mlb_classes = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load mlb_classes.json: {str(e)}")
        raise

    mlb_skin_type_classes = mlb_classes['skin_type']
    mlb_skin_concern_classes = mlb_classes['skin_concern']
    mlb_skin_goal_classes = mlb_classes['skin_goal']
    mlb_ingredient_classes = mlb_classes['ingredients']
    mlb_ingredient_category_classes = mlb_classes['ingredient_category']

    skin_type = clean_input(skin_type)
    skin_concern = clean_input(skin_concern)
    skin_goal = clean_input(skin_goal)
    ingredient = clean_input(ingredient)

    logging.info(f"Normalized inputs: skin_type={skin_type}, skin_concern={skin_concern}, skin_goal={skin_goal}, ingredient={ingredient}")

    unrecognized = [i for i in ingredient if i not in mlb_ingredient_classes]
    if unrecognized:
        logging.warning(f"Some ingredients not recognized: {unrecognized}")

    vec_skin_type = np.array([1 if c in skin_type else 0 for c in mlb_skin_type_classes])
    vec_skin_concern = np.array([1 if c in skin_concern else 0 for c in mlb_skin_concern_classes])
    vec_skin_goal = np.array([1 if c in skin_goal else 0 for c in mlb_skin_goal_classes])
    vec_ingredient = np.array([1 if c in ingredient else 0 for c in mlb_ingredient_classes])

    input_vector = np.concatenate([
        vec_skin_type, vec_skin_concern, vec_skin_goal, vec_ingredient
    ]).reshape(1, -1).astype(np.float32)

    try:
        model = load_model(
            "src/models/ingredients_category_classification_model.keras",
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
        )
    except Exception as e:
        logging.error(f"Failed to load classification model: {str(e)}")
        raise

    probs = model.predict(input_vector, verbose=0)[0]
    logging.info(f"Prediction probabilities: {probs}")

    if use_class_thresholds and best_thresholds is not None:
        if len(best_thresholds) != len(mlb_ingredient_category_classes):
            logging.error(f"Length of best_thresholds ({len(best_thresholds)}) does not match number of categories ({len(mlb_ingredient_category_classes)})")
            raise ValueError("Invalid best_thresholds length")
        predicted_labels = [
            label for label, prob, thres in zip(mlb_ingredient_category_classes, probs, best_thresholds)
            if prob >= thres
        ]
    else:
        predicted_labels = [
            label for label, prob in zip(mlb_ingredient_category_classes, probs)
            if prob >= threshold
        ]

    logging.info(f"Predicted ingredient categories: {predicted_labels}")
    mapped_ingredients = map_category_to_ingredients(predicted_labels)
    logging.info(f"Mapped ingredients: {mapped_ingredients}")

    return mapped_ingredients

# ------------------- LOAD & PREPROCESS ------------------- #
def load_and_prepare_data(file_path):
    logging.info(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

    required_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in dataset: {missing_cols}")
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    try:
        df["review_score"] = df["rating"] * df["total_reviews"]
    except Exception as e:
        logging.error(f"Error calculating review_score: {str(e)}")
        raise

    used_cols = ["product_name", "brand", "image", "price", "rating", "total_reviews", "age", "category", "ingredients", "review_score"]
    df_filtered = df[used_cols].copy()

    def parse_ingredients(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return [x]
        return x if isinstance(x, list) else []
    
    df_filtered["ingredients"] = df_filtered["ingredients"].apply(parse_ingredients)

    logging.info(f"Preprocessing selesai. Total data: {len(df_filtered)}")
    logging.info(f"Sample ingredients: {df_filtered['ingredients'].head().tolist()}")
    logging.info(f"Sample review_score: {df_filtered['review_score'].head().tolist()}")
    logging.info(f"Available columns: {df_filtered.columns.tolist()}")
    logging.info(f"Rating data types: {df['rating'].dtype}, Total Reviews data types: {df['total_reviews'].dtype}")

    return df_filtered

# ------------------- RECOMMENDATION FUNCTION ------------------- #
def recommend_products(df, user_age, user_price_min, user_price_max, user_category, user_ingredients, top_k=5):
    logging.info(f"Mulai filtering rekomendasi berdasarkan input user...")
    logging.info(f"Input user: age={user_age}, category={user_category}, price_range=[{user_price_min}, {user_price_max}], ingredients={user_ingredients}")

    if "review_score" not in df.columns:
        logging.error("review_score column missing in DataFrame")
        raise ValueError("review_score column missing in DataFrame")

    df = df[df["age"] == user_age]
    logging.info(f"After age filter ({user_age}): {len(df)} products")

    df = df[df["category"] == user_category]
    logging.info(f"After category filter ({user_category}): {len(df)} products")

    df = df[(df["price"] >= user_price_min) & (df["price"] <= user_price_max)]
    logging.info(f"After price filter ({user_price_min} - {user_price_max}): {len(df)} products")

    df_before_ingredient_filter = df.copy()
    def has_matching_ingredient(ings):
        if not user_ingredients:
            return True
        normalized_ings = []
        for ing in ings:
            if isinstance(ing, str):
                split_ings = [i.strip().lower() for i in ing.split(",")]
                normalized_ings.extend(split_ings)
            elif isinstance(ing, list):
                normalized_ings.extend([i.lower() for i in ing])
        for user_ing in user_ingredients:
            if user_ing.lower() in normalized_ings:
                return True
        return False

    df = df[df["ingredients"].apply(has_matching_ingredient)]
    logging.info(f"After ingredients filter ({user_ingredients}): {len(df)} products")

    if df.empty and not df_before_ingredient_filter.empty:
        logging.warning("No products match the predicted ingredients. Using products from other filters as fallback.")
        df = df_before_ingredient_filter

    try:
        df_sorted = df.sort_values(by="review_score", ascending=False)
    except Exception as e:
        logging.error(f"Error sorting by review_score: {str(e)}")
        raise

    logging.info(f"Rekomendasi ditemukan: {len(df_sorted)} produk, ambil top-{top_k}")
    return df_sorted.head(top_k)

# ------------------- FLASK ENDPOINT ------------------- #
@app.route('/recommend', methods=['POST'])
def recommend():
    setup_logging()
    file_path = "data/products_integrated_features.csv"

    try:
        # Get JSON input from request
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({"status": "error", "message": "Missing user_input in request body"}), 400

        user_input = data["user_input"]
        required_fields = ["skin_type", "skin_concern", "skin_goal", "ingredient", "age", "price_min", "price_max", "category"]
        missing_fields = [field for field in required_fields if field not in user_input]
        if missing_fields:
            return jsonify({"status": "error", "message": f"Missing required fields: {missing_fields}"}), 400

        best_thresholds = data.get("best_thresholds", [0.5000000000000001, 0.5000000000000001, 0.5000000000000001, 
                                                     0.45000000000000007, 0.5000000000000001, 0.45000000000000007])

        df_filtered = load_and_prepare_data(file_path)

        user_ingredients = predict_ingredient_categories(
            skin_type=user_input["skin_type"],
            skin_concern=user_input["skin_concern"],
            skin_goal=user_input["skin_goal"],
            ingredient=user_input["ingredient"],
            threshold=0.3,
            use_class_thresholds=True,
            best_thresholds=best_thresholds
        )

        top_products = recommend_products(
            df_filtered,
            user_age=user_input["age"],
            user_price_min=user_input["price_min"],
            user_price_max=user_input["price_max"],
            user_category=user_input["category"],
            user_ingredients=user_ingredients,
            top_k=5
        )

        result = {"status": "success", "products": []}
        if not top_products.empty:
            for _, row in top_products.iterrows():
                ingredients = row['ingredients']
                ingredients_str = ', '.join([str(ing) for ing in ingredients if isinstance(ing, str)]) if isinstance(ingredients, list) else str(ingredients)
                result["products"].append({
                    "product_name": row['product_name'],
                    "brand": row['brand'],
                    "image": row['image'],
                    "price": int(row['price']),
                    "rating": float(row['rating']),
                    "total_reviews": int(row['total_reviews']),
                    "ingredients": ingredients_str
                })
        else:
            result["message"] = (
                "Tidak ada produk yang cocok bahkan setelah fallback. Kemungkinan penyebab:\n"
                "- Tidak ada toner untuk umur 25-29 dalam rentang harga tersebut.\n"
                "Saran:\n"
                "- Coba rentang harga lebih luas (misalnya, 0-500000).\n"
                "- Coba skin type, concern, goal, atau ingredient lain.\n"
                "- Periksa kategori lain (misalnya, 'Serum')."
            )

        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Gagal menjalankan sistem: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ------------------- RUN FLASK APP ------------------- #
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)