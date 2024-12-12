from fastapi import APIRouter
from services.recommendation_service import RecommendationService
import pathlib

router = APIRouter()

base_path = pathlib.Path(__file__).parent.parent.resolve()
product_file_path = base_path / "data" / "products.csv"
user_file_path = base_path / "data" / "user_interactions.csv"

recommendation_service = RecommendationService(str(product_file_path), str(user_file_path))


@router.get("/recommendations/{user_id}", tags=["Recommendations"])
def get_recommendations(user_id: int):
    return recommendation_service.get_user_recommendations(user_id)
