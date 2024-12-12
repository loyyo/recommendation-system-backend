import unittest
from fastapi.testclient import TestClient
from main import app


class TestRecommendationServiceFunctional(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_user_recommendations_valid_user(self):
        response = self.client.get("/api/recommendations/1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("product_id", data[0])
        self.assertIn("product_description", data[0])
        self.assertIn("recommendation_value", data[0])

    def test_get_user_recommendations_invalid_user(self):
        response = self.client.get("/api/recommendations/999")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_get_all_products(self):
        response = self.client.get("/api/products")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("product_id", data[0])
        self.assertIn("product_description", data[0])

    def test_get_user_details_valid_id(self):
        response = self.client.get("/api/users/1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("user_id", data[0])

    def test_get_user_details_invalid_id(self):
        response = self.client.get("/api/users/999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data, {"detail": "User with ID 999 not found."})


if __name__ == "__main__":
    unittest.main()
