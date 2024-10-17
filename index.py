from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import uvicorn
import joblib
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Enum classes for input validation
class BodyType(str, Enum):
    UNDERWEIGHT = "Underweight"
    NORMAL = "Normal"
    OVERWEIGHT = "Overweight"
    OBESE = "Obese"


class ScanResult(str, Enum):
    NORMAL = "Normal"
    ABNORMAL = "Abnormal"
    NOT_DONE = "Not done"


class BeforePeriod(str, Enum):
    NONE = "None"
    MOOD_SWINGS = "Mood swings"
    BLOATING = "Bloating"
    CONSTIPATION = "Constipation"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


# Input model with validation
class PCOSInput(BaseModel):
    age: str = Field(..., description="Age range (e.g., '19-34')")
    body_type: BodyType
    scanning: ScanResult
    before_period: BeforePeriod
    irregular_period: YesNo
    painful_period: YesNo
    bleeding: YesNo
    exercise: YesNo
    Hereditary: YesNo = Field(alias="Hereditary")
    diabetes: YesNo
    hypothyroidism: YesNo
    hair_growth: YesNo
    acne: YesNo

    @field_validator('age')
    def validate_age(cls, v):
        if not v.replace('-', '').isdigit():
            raise ValueError('Age must be in format "XX-YY"')
        return v


class PCOSPredictorAPI:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'body_type', 'scanning', 'before_period', 'irregular_period',
            'painful_period', 'bleeding', 'exercise', 'Hereditary', 'diabetes',
            'hypothyroidism', 'hair_growth', 'acne'
        ]
        self.initialize_model()

    def initialize_model(self):
        """Initialize and train the model with default data."""
        # Training data using the exact enum values
        training_data = pd.DataFrame({
            'age': ['19-34'] * 5,
            'body_type': [BodyType.NORMAL.value, BodyType.OVERWEIGHT.value, BodyType.NORMAL.value,
                          BodyType.OBESE.value, BodyType.NORMAL.value],
            'scanning': [ScanResult.NORMAL.value, ScanResult.NORMAL.value, ScanResult.ABNORMAL.value,
                         ScanResult.NORMAL.value, ScanResult.NORMAL.value],
            'before_period': [BeforePeriod.NONE.value, BeforePeriod.MOOD_SWINGS.value,
                              BeforePeriod.BLOATING.value, BeforePeriod.NONE.value,
                              BeforePeriod.MOOD_SWINGS.value],
            'irregular_period': [YesNo.NO.value, YesNo.YES.value, YesNo.YES.value,
                                 YesNo.NO.value, YesNo.NO.value],
            'painful_period': [YesNo.NO.value, YesNo.YES.value, YesNo.YES.value,
                               YesNo.NO.value, YesNo.YES.value],
            'bleeding': [YesNo.NO.value, YesNo.YES.value, YesNo.NO.value,
                         YesNo.NO.value, YesNo.YES.value],
            'exercise': [YesNo.YES.value, YesNo.NO.value, YesNo.NO.value,
                         YesNo.YES.value, YesNo.NO.value],
            'Hereditary': [YesNo.NO.value, YesNo.YES.value, YesNo.NO.value,
                           YesNo.NO.value, YesNo.YES.value],
            'diabetes': [YesNo.NO.value, YesNo.NO.value, YesNo.NO.value,
                         YesNo.YES.value, YesNo.NO.value],
            'hypothyroidism': [YesNo.NO.value, YesNo.NO.value, YesNo.YES.value,
                               YesNo.NO.value, YesNo.NO.value],
            'hair_growth': [YesNo.NO.value, YesNo.YES.value, YesNo.YES.value,
                            YesNo.NO.value, YesNo.YES.value],
            'acne': [YesNo.NO.value, YesNo.YES.value, YesNo.YES.value,
                     YesNo.NO.value, YesNo.NO.value]
        })
        target = pd.Series([0, 1, 1, 0, 1])

        # Initialize label encoders with all possible values
        self.initialize_label_encoders()

        X = self.preprocess_input(training_data)
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, target)

    def initialize_label_encoders(self):
        """Initialize label encoders with all possible values for each categorical feature."""
        categorical_features = {
            'body_type': [e.value for e in BodyType],
            'scanning': [e.value for e in ScanResult],
            'before_period': [e.value for e in BeforePeriod],
            'irregular_period': [e.value for e in YesNo],
            'painful_period': [e.value for e in YesNo],
            'bleeding': [e.value for e in YesNo],
            'exercise': [e.value for e in YesNo],
            'Hereditary': [e.value for e in YesNo],
            'diabetes': [e.value for e in YesNo],
            'hypothyroidism': [e.value for e in YesNo],
            'hair_growth': [e.value for e in YesNo],
            'acne': [e.value for e in YesNo]
        }

        for feature, values in categorical_features.items():
            self.label_encoders[feature] = LabelEncoder()
            self.label_encoders[feature].fit(values)

    def preprocess_input(self, data):
        """Preprocess input data."""
        if isinstance(data, dict):
            # Convert enum values to their string representations
            processed_dict = {}
            for key, value in data.items():
                if isinstance(value, Enum):
                    processed_dict[key] = value.value
                else:
                    processed_dict[key] = value
            data = pd.DataFrame([processed_dict])

        processed_data = data.copy()

        # Transform categorical features using pre-initialized label encoders
        for feature in self.label_encoders.keys():
            if feature in processed_data.columns:
                processed_data[feature] = self.label_encoders[feature].transform(processed_data[feature])

        # Process age feature
        if 'age' in processed_data.columns:
            processed_data['age'] = processed_data['age'].apply(lambda x: int(str(x).split('-')[0]))

        return processed_data[self.feature_columns]

    def predict(self, input_data: dict) -> dict:
        """Make prediction and return results."""
        try:
            processed_input = self.preprocess_input(input_data)
            scaled_input = self.scaler.transform(processed_input)

            prediction = self.model.predict(scaled_input)[0]
            probability = self.model.predict_proba(scaled_input)[0][1]

            risk_level = "Low" if probability < 0.4 else "Medium" if probability < 0.7 else "High"

            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            key_factors = [k for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]]

            return {
                "prediction": bool(prediction),
                "probability": float(probability),
                "risk_level": risk_level,
                "key_factors": key_factors
            }
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="PCOS Prediction API",
    description="API for predicting PCOS risk based on patient symptoms and characteristics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = PCOSPredictorAPI()


@app.post("/predict")
async def predict_pcos(input_data: PCOSInput):
    try:
        prediction = predictor.predict(input_data.dict(by_alias=True))
        return prediction
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)