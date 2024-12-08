from fastapi import APIRouter
from services.product_service import ProductService

router = APIRouter()
product_service = ProductService("data/products.csv")


@router.get("/products", tags=["Products"])
def get_products():
    return product_service.get_products()
