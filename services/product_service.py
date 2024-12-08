import pandas as pd


class ProductService:
    def __init__(self, product_file):
        self.product_data = pd.read_csv(product_file)

    def get_products(self):
        return self.product_data.to_dict(orient="records")
