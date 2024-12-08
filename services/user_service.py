import pandas as pd
from fastapi import HTTPException


class UserService:
    def __init__(self, user_file):
        self.user_data = pd.read_csv(user_file)
        
    def get_user_ids(self):
        user_ids = self.user_data["user_id"].unique().tolist()
        return {"user_ids": user_ids}
    
    def get_user_data(self, user_id):
        user_interactions = self.user_data[self.user_data["user_id"] == user_id]
        if user_interactions.empty:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found.")
        return user_interactions.to_dict(orient="records")
