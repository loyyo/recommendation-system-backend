from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import user_controller, product_controller, recommendation_controller

app = FastAPI(title="Recommendation System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_controller.router, prefix="/api")
app.include_router(product_controller.router, prefix="/api")
app.include_router(recommendation_controller.router, prefix="/api")