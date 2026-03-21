from pydantic import BaseModel, Field, ConfigDict
class PredictResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    predicted_class: str = Field(..., description="The label of the item")
    confidence: float = Field(..., ge=0, le=1, description="Probability score between 0 and 1")
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra = "forbid",
        json_schema_extra={
            "examples": [
                {
                    "filename": "test_image.png",
                    "predicted_class": "Ankle boot",
                    "confidence": 0.77
                }
            ]
        }
    )