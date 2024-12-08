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
		"""Validate input data for required columns and non-empty datasets."""
		required_product_columns = {"product_id", "product_description"}
		required_user_columns = {"user_id", "product_id", "interaction"}
		
		if self.product_data.empty or self.user_data.empty:
			raise ValueError("Input data is empty. Check the input files.")
		
		if not required_product_columns.issubset(self.product_data.columns):
			raise ValueError(f"Product data must contain columns: {required_product_columns}")
		
		if not required_user_columns.issubset(self.user_data.columns):
			raise ValueError(f"User data must contain columns: {required_user_columns}")
	
	def _analyze_products(self):
		"""Compute product similarity matrix using TF-IDF."""
		vectorizer = TfidfVectorizer(stop_words="english")
		tfidf_matrix = vectorizer.fit_transform(self.product_data["product_description"])
		self.product_similarity_matrix = cosine_similarity(tfidf_matrix)
	
	def _analyze_users(self):
		"""Compute user-item matrix and user similarity matrix."""
		self.user_item_matrix = self.user_data.pivot_table(
			index="user_id", columns="product_id", values="interaction", fill_value=0
		)
		self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
	
	def get_user_recommendations(self, user_id):
		"""Generate top recommendations for a user."""
		if user_id not in self.user_item_matrix.index:
			return {"error": f"User {user_id} not found."}
		
		cb_recs = self._generate_cb_recommendations(user_id)
		cf_recs = self._generate_cf_recommendations(user_id)
		combined_recs = self._merge_recommendations(cb_recs, cf_recs)
		
		return combined_recs.head(5).to_dict(orient="records")
	
	def _generate_cb_recommendations(self, user_id):
		"""Generate content-based recommendations."""
		user_history = self.user_data[self.user_data["user_id"] == user_id]
		if user_history.empty:
			return pd.DataFrame()
		
		# Optimize iterative operations using list comprehension
		recommendations = pd.concat(
			[self._get_similar_products(product_id) for product_id in user_history["product_id"]],
			ignore_index=True
		)
		recommendations = recommendations[~recommendations["product_id"].isin(user_history["product_id"])]
		recommendations["cb_score"] = np.linspace(1, 0.5, len(recommendations))
		recommendations["product_description"] = recommendations["product_id"].map(
			self.product_data.set_index("product_id")["product_description"]
		)
		return recommendations
	
	def _generate_cf_recommendations(self, user_id):
		"""Generate collaborative filtering recommendations."""
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
		"""Merge content-based and collaborative filtering recommendations."""
		if cb_recs.empty and cf_recs.empty:
			return pd.DataFrame(columns=["product_id", "product_description", "recommendation_value"])
		
		# Set indices for merging
		cb_recs = cb_recs.set_index("product_id", drop=False)
		cf_recs = cf_recs.set_index("product_id", drop=False)
		
		# Merge the two DataFrames
		merged = cb_recs.join(cf_recs, how="outer", lsuffix="_cb", rsuffix="_cf").fillna(0)
		
		# Debugging: Check column names after join
		if "product_id_cb" not in merged.columns or "product_id_cf" not in merged.columns:
			raise ValueError(f"Expected columns missing in merged DataFrame: {merged.columns}")
		
		# Resolve column conflicts and ensure product_id is correctly mapped
		merged["product_id"] = np.where(
			merged["product_id_cb"] == 0,
			merged["product_id_cf"],
			merged["product_id_cb"]
		)
		merged["product_description"] = np.where(
			merged["product_description_cb"] == 0,
			merged["product_description_cf"],
			merged["product_description_cb"]
		)
		merged["recommendation_value"] = (merged.get("cb_score", 0) + merged.get("cf_score", 0)) / 2
		
		# Reset index and retain only required columns
		required_columns = ["product_id", "product_description", "recommendation_value"]
		merged = merged.reset_index(drop=True)
		
		# Check for required columns
		if not set(required_columns).issubset(merged.columns):
			raise ValueError(f"Missing required columns in merged DataFrame: {merged.columns}")
		
		return merged[required_columns]
	
	def _get_similar_products(self, product_id, top_n=5):
		"""Find top-N similar products."""
		if product_id not in self.product_data["product_id"].values:
			return pd.DataFrame(columns=["product_id", "product_description"])
		
		product_idx = self.product_data[self.product_data["product_id"] == product_id].index[0]
		similarity_scores = self.product_similarity_matrix[product_idx]
		similar_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
		
		similar_products = self.product_data.iloc[similar_indices].copy()
		return similar_products.reset_index(drop=True)
