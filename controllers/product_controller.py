from fastapi import APIRouter
from services.product_service import ProductService
import pathlib

router = APIRouter()

base_path = pathlib.Path(__file__).parent.parent.resolve()
product_file_path = base_path / "data" / "products.csv"

product_service = ProductService(str(product_file_path))


@router.get("/products", tags=["Products"])
def get_products():
    return product_service.get_products()
