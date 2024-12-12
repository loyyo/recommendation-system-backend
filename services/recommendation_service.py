import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RecommendationService:
    def __init__(self, product_file, user_file):
        try:
            self.product_data = pd.read_csv(product_file)
            self.user_data = pd.read_csv(user_file)
        except FileNotFoundError as e:
            raise ValueError(f"Error loading files: {e}")
        except pd.errors.EmptyDataError:
            raise ValueError("One of the input files is empty.")

        self.user_item_matrix = None
        self.product_similarity_matrix = None
        self.user_similarity_matrix = None

        self._validate_data()
        self._analyze_products()
        self._analyze_users()

    def _validate_data(self):
        required_product_columns = {"product_id", "product_description"}
        required_user_columns = {"user_id", "product_id", "interaction"}

        if self.product_data.empty or self.user_data.empty:
            raise ValueError("Input data is empty. Check the input files.")

        if not required_product_columns.issubset(self.product_data.columns):
            raise ValueError(f"Product data must contain columns: {required_product_columns}")

        if not required_user_columns.issubset(self.user_data.columns):
            raise ValueError(f"User data must contain columns: {required_user_columns}")

    def _analyze_products(self):
        if self.product_data.empty:
            return

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(self.product_data["product_description"])
        self.product_similarity_matrix = cosine_similarity(tfidf_matrix)

    def _analyze_users(self):
        self.user_item_matrix = self.user_data.pivot_table(
            index="user_id", columns="product_id", values="interaction", fill_value=0
        )
        if not self.user_item_matrix.empty:
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)

    def get_user_recommendations(self, user_id):
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []

        cb_recs = self._generate_cb_recommendations(user_id)
        cf_recs = self._generate_cf_recommendations(user_id)
        combined_recs = self._merge_recommendations(cb_recs, cf_recs)

        return combined_recs.head(5).to_dict(orient="records")

    def _generate_cb_recommendations(self, user_id):
        user_history = self.user_data[self.user_data["user_id"] == user_id]

        if user_history.empty:
            return pd.DataFrame(columns=["product_id", "product_description", "similarity", "cb_score"])

        recommendations = []
        for product_id in user_history["product_id"]:
            similar_products = self._get_similar_products(product_id)
            recommendations.append(similar_products)

        if recommendations:
            recommendations = pd.concat(recommendations).drop_duplicates(subset="product_id")

            # Ensure consistent types for filtering
            interacted_products = user_history["product_id"].dropna().astype(str).tolist()
            recommendations["product_id"] = recommendations["product_id"].astype(str)

            filtered_recommendations = recommendations[~recommendations["product_id"].isin(interacted_products)]

            if filtered_recommendations.empty:
                filtered_recommendations = recommendations.head(5)

            if not filtered_recommendations.empty:
                filtered_recommendations["cb_score"] = np.linspace(1, 0.5, len(filtered_recommendations))
            return filtered_recommendations.reset_index(drop=True)
        else:
            return pd.DataFrame(columns=["product_id", "product_description", "similarity", "cb_score"])

    def _generate_cf_recommendations(self, user_id):
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix is not initialized.")

        user_idx = list(self.user_item_matrix.index).index(user_id)
        similarity_scores = self.user_similarity_matrix[user_idx]
        similar_users_indices = np.argsort(similarity_scores)[::-1][1:]
        similar_users = self.user_item_matrix.index[similar_users_indices]
        similar_users_data = self.user_item_matrix.loc[similar_users]

        mean_ratings = similar_users_data.mean(axis=0)
        user_interactions = self.user_item_matrix.loc[user_id]
        products_to_recommend = mean_ratings[user_interactions == 0].sort_values(ascending=False)

        recommended_products = self.product_data[
            self.product_data["product_id"].isin(products_to_recommend.index)
        ].copy()
        recommended_products["cf_score"] = products_to_recommend.loc[
            recommended_products["product_id"]
        ].values
        return recommended_products

    def _merge_recommendations(self, cb_recs, cf_recs):
        if cb_recs.empty and cf_recs.empty:
            return pd.DataFrame(columns=["product_id", "product_description", "recommendation_value"])

        cb_recs = cb_recs.set_index("product_id", drop=False) if not cb_recs.empty else pd.DataFrame(
            columns=["product_id", "product_description", "cb_score"])
        cf_recs = cf_recs.set_index("product_id", drop=False) if not cf_recs.empty else pd.DataFrame(
            columns=["product_id", "product_description", "cf_score"])

        merged = cb_recs.join(cf_recs, how="outer", lsuffix="_cb", rsuffix="_cf").fillna(0)

        product_id_cb = merged.get("product_id_cb", pd.Series(dtype=int))
        product_id_cf = merged.get("product_id_cf", pd.Series(dtype=int))
        merged["product_id"] = product_id_cb.combine_first(product_id_cf).astype(int)

        description_cb = merged.get("product_description_cb", pd.Series(dtype=str))
        description_cf = merged.get("product_description_cf", pd.Series(dtype=str))
        merged["product_description"] = description_cb.combine_first(description_cf)

        merged["recommendation_value"] = (merged.get("cb_score", 0) + merged.get("cf_score", 0)) / 2

        return merged.reset_index(drop=True)[["product_id", "product_description", "recommendation_value"]]

    def _get_similar_products(self, product_id, top_n=5):
        if product_id not in self.product_data["product_id"].values:
            return pd.DataFrame(columns=["product_id", "product_description"])

        product_idx = self.product_data[self.product_data["product_id"] == product_id].index[0]
        similarity_scores = self.product_similarity_matrix[product_idx]
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]

        similar_products = self.product_data.iloc[similar_indices].copy()
        similar_products["similarity"] = similarity_scores[similar_indices]
        return similar_products.reset_index(drop=True)
