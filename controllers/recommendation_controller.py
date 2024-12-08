from fastapi import APIRouter
from services.recommendation_service import RecommendationService

router = APIRouter()
recommendation_service = RecommendationService("data/products.csv", "data/user_interactions.csv")


@router.get("/recommendations/{user_id}", tags=["Recommendations"])
def get_recommendations(user_id: int):
    return recommendation_service.get_user_recommendations(user_id)
