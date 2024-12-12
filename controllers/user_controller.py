from fastapi import APIRouter
from services.user_service import UserService
import pathlib

router = APIRouter()

base_path = pathlib.Path(__file__).parent.parent.resolve()
user_file_path = base_path / "data" / "user_interactions.csv"

user_service = UserService(str(user_file_path))


@router.get("/users/ids", tags=["Users"])
def get_user_ids():
    return user_service.get_user_ids()


@router.get("/users/{user_id}", tags=["User"])
def get_user_data(user_id: int):
    return user_service.get_user_data(user_id)
