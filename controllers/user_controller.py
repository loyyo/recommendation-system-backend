from fastapi import APIRouter
from services.user_service import UserService

router = APIRouter()
user_service = UserService("data/user_interactions.csv")


@router.get("/users/ids", tags=["Users"])
def get_user_ids():
    return user_service.get_user_ids()


@router.get("/users/{user_id}", tags=["User"])
def get_user_data(user_id: int):
    return user_service.get_user_data(user_id)
