import unittest
import pandas as pd
from tempfile import NamedTemporaryFile
from services.recommendation_service import RecommendationService


class TestRecommendationServiceUnit(unittest.TestCase):
    def setUp(self):
        self.temp_product_file = NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.temp_user_file = NamedTemporaryFile(mode='w', delete=False, suffix='.csv')

        self.temp_product_file.write("product_id,product_description\n1,Product A\n2,Product B\n3,Product C\n")
        self.temp_user_file.write("user_id,product_id,interaction\n1,1,5\n1,2,3\n2,3,4\n")
        self.temp_product_file.close()
        self.temp_user_file.close()

        self.service = RecommendationService(self.temp_product_file.name, self.temp_user_file.name)

    def tearDown(self):
        import os
        os.unlink(self.temp_product_file.name)
        os.unlink(self.temp_user_file.name)

    def test_validate_data(self):
        try:
            self.service._validate_data()
        except Exception as e:
            self.fail(f"Validation raised an exception unexpectedly: {e}")

    def test_validate_data_missing_columns(self):
        with NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as invalid_file:
            invalid_file.write("product_id\n1\n2\n3\n")
            invalid_file.close()

        with self.assertRaises(ValueError):
            RecommendationService(invalid_file.name, self.temp_user_file.name)

    def test_analyze_products(self):
        self.service._analyze_products()
        self.assertIsNotNone(self.service.product_similarity_matrix)

    def test_analyze_users(self):
        self.service._analyze_users()
        self.assertIsNotNone(self.service.user_item_matrix)
        self.assertIsNotNone(self.service.user_similarity_matrix)

    def test_generate_cb_recommendations(self):
        cb_recs = self.service._generate_cb_recommendations(1)
        self.assertFalse(cb_recs.empty, "Content-based recommendations should not be empty.")
        self.assertIn("product_id", cb_recs.columns, "Expected 'product_id' column in recommendations.")
        self.assertIn("cb_score", cb_recs.columns, "Expected 'cb_score' column in recommendations.")

    def test_generate_cb_recommendations_no_history(self):
        cb_recs = self.service._generate_cb_recommendations(999)
        self.assertTrue(cb_recs.empty, "Recommendations should be empty for a user with no history.")

    def test_generate_cf_recommendations(self):
        cf_recs = self.service._generate_cf_recommendations(1)
        self.assertFalse(cf_recs.empty, "Collaborative filtering recommendations should not be empty.")

    def test_generate_cf_recommendations_no_similar_users(self):
        self.service.user_similarity_matrix = None
        with self.assertRaises(ValueError):
            self.service._generate_cf_recommendations(1)

    def test_merge_recommendations(self):
        cb_recs = self.service._generate_cb_recommendations(1)
        cf_recs = self.service._generate_cf_recommendations(1)
        merged_recs = self.service._merge_recommendations(cb_recs, cf_recs)
        self.assertFalse(merged_recs.empty, "Merged recommendations should not be empty.")
        self.assertIn("product_id", merged_recs.columns)
        self.assertIn("product_description", merged_recs.columns)
        self.assertIn("recommendation_value", merged_recs.columns)

    def test_merge_recommendations_empty(self):
        empty_cb_recs = pd.DataFrame(columns=["product_id", "product_description", "cb_score"])
        empty_cf_recs = pd.DataFrame(columns=["product_id", "product_description", "cf_score"])
        merged_recs = self.service._merge_recommendations(empty_cb_recs, empty_cf_recs)
        self.assertTrue(merged_recs.empty, "Merged recommendations should be empty.")
        self.assertIn("product_id", merged_recs.columns)
        self.assertIn("product_description", merged_recs.columns)
        self.assertIn("recommendation_value", merged_recs.columns)

    def test_get_similar_products(self):
        similar_products = self.service._get_similar_products(1)
        self.assertFalse(similar_products.empty, "Similar products should not be empty.")

    def test_get_similar_products_invalid_id(self):
        similar_products = self.service._get_similar_products(999)
        self.assertTrue(similar_products.empty, "Similar products should be empty for an invalid product ID.")

    def test_get_user_recommendations_valid_user(self):
        recommendations = self.service.get_user_recommendations(1)
        self.assertTrue(isinstance(recommendations, list), "Recommendations should be a list.")
        self.assertGreater(len(recommendations), 0, "Recommendations list should not be empty.")

    def test_get_user_recommendations_invalid_user(self):
        recommendations = self.service.get_user_recommendations(999)
        self.assertEqual(len(recommendations), 0, "Recommendations should be empty for an invalid user.")


if __name__ == "__main__":
    unittest.main()
