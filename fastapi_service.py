from __future__ import annotations

import io
from typing import List, Optional, Annotated

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile

import pickle

TARGET_COL = "selling_price"


class JupyterNotebookUnpickler(pickle.Unpickler):
    def find_class(self, module_name, global_name):
        if module_name == "__main__":
            import custom_sklearn_transformers
            return super().find_class("custom_sklearn_transformers", global_name)
        return super().find_class(module_name, global_name)


class Model:
    def __init__(self, pkl_model_filename: str, pkl_full_pipeline_filename: str):
        with open(pkl_model_filename, "rb") as f:
            self.sklearn_model = JupyterNotebookUnpickler(f).load()
        with open(pkl_full_pipeline_filename, "rb") as f:
            self.sklearn_pipeline = JupyterNotebookUnpickler(f).load()

    def _predict(self, data: pd.DataFrame):
        return self.sklearn_model.predict(self.sklearn_pipeline.transform(data))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.drop([TARGET_COL], axis=1, errors='ignore')
        df[TARGET_COL] = self._predict(data)
        return df


model = Model(
    pkl_model_filename="pickle/best_model_cat_ridge.pkl",
    pkl_full_pipeline_filename="pickle/full_pipeline_cat.pkl",
)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: Optional[int] = Field(default=None)
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return model.predict(pd.DataFrame.from_records([item.model_dump()]))[TARGET_COL][0]


@app.post("/predict_items")
def predict_items(read: Annotated[bytes, File(alias="data")]) -> StreamingResponse:
    df = pd.read_csv(io.StringIO(read.decode("UTF-8")))
    df = model.predict(df)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
