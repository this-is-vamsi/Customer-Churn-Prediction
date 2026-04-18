"""
FastAPI Application for Customer Churn Prediction
==================================================

This is a REST API that serves our trained Random Forest model to predict
whether a customer will churn (leave) or not.

What is FastAPI?
----------------
FastAPI is a modern, fast web framework for building APIs with Python.
It's built on top of Starlette (for web routing) and Pydantic (for data validation).

Key Features:
- Fast: High performance, comparable to NodeJS and Go
- Easy: Simple to use and learn
- Automatic documentation: Creates interactive API docs automatically
- Type hints: Uses Python type hints for validation
- Async support: Can handle async operations efficiently
"""

# ============================================================================
# IMPORTS SECTION
# ============================================================================

# FastAPI is the main framework we use to create our API
from fastapi import FastAPI, HTTPException

# Pydantic is used for data validation and creating data models
# BaseModel: A class that helps us define the structure of our input/output data
from pydantic import BaseModel, Field

# Type hints help specify what type of data we expect
from typing import List, Dict

# joblib: Used to load our saved machine learning model and encoders
import joblib

# json: Used to read JSON files (feature names, encoder info)
import json

# numpy: Used for numerical operations
import numpy as np

# pandas: Used for data manipulation
import pandas as pd

# os: Used for file path operations
import os


# ============================================================================
# APP INITIALIZATION
# ============================================================================

# Create the FastAPI application instance
# Think of this as creating your web server
app = FastAPI(
    title="Customer Churn Prediction API",  # API name shown in documentation
    description="API to predict customer churn using Random Forest model",  # API description
    version="1.0.0"  # Version number for tracking
)

# What does 'app' do?
# -------------------
# 'app' is the main FastAPI application object. It:
# 1. Handles incoming HTTP requests (GET, POST, etc.)
# 2. Routes requests to the correct functions
# 3. Manages response generation
# 4. Handles errors and exceptions
# 5. Automatically generates API documentation


# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================

# Define the path where our model and related files are stored
MODEL_PATH = "model"

# Load the trained Random Forest model
# joblib.load() reads the saved model from disk into memory
model = joblib.load(os.path.join(MODEL_PATH, "best_churn_model.pkl"))

# Load the label encoders (these convert categorical text to numbers)
# We need these to transform Geography and Gender inputs
geography_encoder = joblib.load(os.path.join(MODEL_PATH, "geography_encoder.pkl"))
gender_encoder = joblib.load(os.path.join(MODEL_PATH, "gender_encoder.pkl"))

# Load feature names (the order of features the model expects)
with open(os.path.join(MODEL_PATH, "feature_names.json"), 'r') as f:
    feature_names = json.load(f)

# Load encoder information (mapping of categories to numbers)
with open(os.path.join(MODEL_PATH, "encoder_info.json"), 'r') as f:
    encoder_info = json.load(f)

# Why do we load these at startup?
# ---------------------------------
# Loading the model once when the app starts is much faster than loading it
# every time someone makes a prediction request. This improves API performance.


# ============================================================================
# PYDANTIC MODELS (DATA VALIDATION SCHEMAS)
# ============================================================================

"""
What is Pydantic and BaseModel?
--------------------------------
Pydantic is a data validation library that uses Python type hints.

BaseModel is a class from Pydantic that:
1. Defines the structure of data (what fields are required)
2. Validates incoming data automatically (checks types, ranges, etc.)
3. Converts data to the correct types
4. Generates JSON schemas for documentation
5. Provides clear error messages when data is invalid

Why use Pydantic?
-----------------
- Automatic validation: No need to write manual validation code
- Type safety: Ensures data has the correct type
- Clear errors: Tells users exactly what's wrong with their input
- Documentation: Automatically documents what data the API expects
"""


class CustomerData(BaseModel):
    """
    Input Model - Defines the structure of customer data we expect
    
    This class inherits from BaseModel, which gives it superpowers:
    - Automatic validation of all fields
    - Automatic conversion to correct types
    - Clear error messages for invalid data
    
    Each field uses Field() to add extra validation and documentation:
    - description: Explains what the field is for
    - ge: "greater than or equal" - minimum value
    - le: "less than or equal" - maximum value
    - example: Shows an example value in API docs
    """
    
    # Credit Score: Must be between 300 and 850
    # The '...' means this field is REQUIRED (user must provide it)
    CreditScore: int = Field(
        ...,  # Required field
        description="Customer's credit score",
        ge=300,  # Must be >= 300
        le=850,  # Must be <= 850
        example=650
    )
    
    # Geography: Must be one of the three countries
    # We use str type and validate it's a known country
    Geography: str = Field(
        ...,
        description="Customer's country (France, Spain, or Germany)",
        example="France"
    )
    
    # Gender: Must be Male or Female
    Gender: str = Field(
        ...,
        description="Customer's gender (Male or Female)",
        example="Female"
    )
    
    # Age: Must be between 18 and 100
    Age: int = Field(
        ...,
        description="Customer's age",
        ge=18,
        le=100,
        example=42
    )
    
    # Tenure: Years with bank, between 0 and 10
    Tenure: int = Field(
        ...,
        description="Number of years as a customer",
        ge=0,
        le=10,
        example=5
    )
    
    # Balance: Account balance, must be non-negative
    Balance: float = Field(
        ...,
        description="Account balance",
        ge=0,
        example=125000.00
    )
    
    # NumOfProducts: Number of products, between 1 and 4
    NumOfProducts: int = Field(
        ...,
        description="Number of bank products",
        ge=1,
        le=4,
        example=2
    )
    
    # HasCrCard: 0 or 1 (boolean as integer)
    HasCrCard: int = Field(
        ...,
        description="Has credit card (0 = No, 1 = Yes)",
        ge=0,
        le=1,
        example=1
    )
    
    # IsActiveMember: 0 or 1 (boolean as integer)
    IsActiveMember: int = Field(
        ...,
        description="Is active member (0 = No, 1 = Yes)",
        ge=0,
        le=1,
        example=1
    )
    
    # EstimatedSalary: Must be non-negative
    EstimatedSalary: float = Field(
        ...,
        description="Estimated annual salary",
        ge=0,
        example=100000.00
    )
    
    # Why use this model?
    # -------------------
    # When a user sends data to our API, FastAPI automatically:
    # 1. Checks if all required fields are present
    # 2. Validates that each field has the correct type
    # 3. Checks that numbers are within the specified ranges
    # 4. Returns a clear error message if anything is wrong
    # 5. Converts the data to a Python object we can use


class PredictionResponse(BaseModel):
    """
    Output Model - Defines the structure of our prediction response
    
    This tells FastAPI what our API will return to the user.
    """
    
    # Will the customer churn? 0 = No, 1 = Yes
    prediction: int = Field(
        ...,
        description="Churn prediction (0 = No Churn, 1 = Churn)"
    )
    
    # How confident is the model? (probability from 0 to 1)
    churn_probability: float = Field(
        ...,
        description="Probability of customer churning (0.0 to 1.0)"
    )
    
    # Human-readable interpretation
    risk_level: str = Field(
        ...,
        description="Risk level (Low, Medium, High)"
    )
    
    # Why use this model?
    # -------------------
    # - Ensures our API always returns data in a consistent format
    # - Automatically generates documentation showing what the API returns
    # - Type-safe: Ensures we don't accidentally return wrong data types


class BatchPredictionRequest(BaseModel):
    """
    Batch Input Model - For predicting multiple customers at once
    
    This allows users to send multiple customer records in a single request.
    """
    
    customers: List[CustomerData] = Field(
        ...,
        description="List of customer data for batch prediction"
    )
    
    # List[CustomerData] means: A list where each item is a CustomerData object
    # This enables bulk predictions, which is more efficient than individual requests


class BatchPredictionResponse(BaseModel):
    """
    Batch Output Model - Returns predictions for multiple customers
    """
    
    predictions: List[Dict] = Field(
        ...,
        description="List of predictions for each customer"
    )
    
    total_customers: int = Field(
        ...,
        description="Total number of customers processed"
    )
    
    total_churn: int = Field(
        ...,
        description="Number of customers predicted to churn"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_input(customer_data: CustomerData) -> pd.DataFrame:
    """
    Preprocesses customer data for model prediction
    
    This function:
    1. Validates that categorical values are known
    2. Encodes categorical variables (Geography, Gender) to numbers
    3. Arranges features in the correct order
    4. Converts to a DataFrame (the format our model expects)
    
    Parameters:
    -----------
    customer_data : CustomerData
        The customer data from the API request
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with one row, ready for model prediction
    
    Raises:
    -------
    HTTPException
        If Geography or Gender values are not recognized
    """
    
    # Step 1: Validate Geography
    # Check if the provided country is one we know about
    if customer_data.Geography not in geography_encoder.classes_:
        # If not, raise an HTTP error with status code 400 (Bad Request)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Geography. Must be one of: {list(geography_encoder.classes_)}"
        )
    
    # Step 2: Validate Gender
    # Check if the provided gender is one we know about
    if customer_data.Gender not in gender_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Gender. Must be one of: {list(gender_encoder.classes_)}"
        )
    
    # Step 3: Encode categorical variables
    # Transform text values to numbers using our saved encoders
    # transform() returns an array, [0] gets the first (and only) element
    geography_encoded = geography_encoder.transform([customer_data.Geography])[0]
    gender_encoded = gender_encoder.transform([customer_data.Gender])[0]
    
    # Step 4: Create feature dictionary
    # Build a dictionary with all features in the correct order
    # The model was trained with features in a specific order, so we must match it
    features = {
        'CreditScore': customer_data.CreditScore,
        'Geography': geography_encoded,  # Now it's a number
        'Gender': gender_encoded,        # Now it's a number
        'Age': customer_data.Age,
        'Tenure': customer_data.Tenure,
        'Balance': customer_data.Balance,
        'NumOfProducts': customer_data.NumOfProducts,
        'HasCrCard': customer_data.HasCrCard,
        'IsActiveMember': customer_data.IsActiveMember,
        'EstimatedSalary': customer_data.EstimatedSalary
    }
    
    # Step 5: Convert to DataFrame
    # Create a DataFrame with one row
    # The model expects a DataFrame, not a dictionary
    df = pd.DataFrame([features])
    
    # Step 6: Ensure correct feature order
    # Reorder columns to match the training data
    df = df[feature_names]
    
    return df


def get_risk_level(probability: float) -> str:
    """
    Converts churn probability to a risk level category
    
    This makes the prediction easier to understand for business users.
    
    Parameters:
    -----------
    probability : float
        Churn probability from 0.0 to 1.0
    
    Returns:
    --------
    str
        Risk level: "Low", "Medium", or "High"
    """
    
    if probability < 0.3:
        return "Low"      # Less than 30% chance of churning
    elif probability < 0.6:
        return "Medium"   # 30-60% chance of churning
    else:
        return "High"     # More than 60% chance of churning


# ============================================================================
# API ENDPOINTS (ROUTES)
# ============================================================================

"""
What is an API Endpoint?
------------------------
An endpoint is a specific URL where your API can receive requests.
Each endpoint is like a function that users can call over the internet.

Decorators (@app.get, @app.post) tell FastAPI:
- What URL this function responds to
- What HTTP method to use (GET, POST, etc.)
- What the function does (via summary and description)
"""


@app.get("/")
def root():
    """
    Root endpoint - Welcome message
    
    What is @app.get("/")?
    ----------------------
    - @app.get: This is a decorator that tells FastAPI this function handles GET requests
    - "/": This is the URL path (the root of our API)
    - When someone visits http://your-api.com/, this function runs
    
    GET vs POST:
    ------------
    - GET: Used to retrieve information (like viewing a webpage)
    - POST: Used to send data to the server (like submitting a form)
    
    Returns:
    --------
    A dictionary that FastAPI automatically converts to JSON
    """
    
    return {
        "message": "Customer Churn Prediction API",
        "status": "active",
        "model": "Random Forest (Tuned)",
        "endpoints": {
            "predict": "/predict - Single customer prediction",
            "batch_predict": "/batch-predict - Multiple customers prediction",
            "health": "/health - API health check",
            "model_info": "/model-info - Model details"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    
    This endpoint is used to check if the API is running properly.
    It's commonly used by monitoring systems and load balancers.
    
    Returns:
    --------
    A simple status message
    """
    
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    """
    Model information endpoint
    
    Returns information about the trained model.
    Useful for users who want to know what model they're using.
    
    Returns:
    --------
    Dictionary with model details
    """
    
    return {
        "model_type": "Random Forest Classifier",
        "status": "Tuned with Grid Search CV",
        "features": feature_names,
        "geography_options": list(geography_encoder.classes_),
        "gender_options": list(gender_encoder.classes_),
        "geography_mapping": encoder_info['geography_mapping'],
        "gender_mapping": encoder_info['gender_mapping'],
        "performance_metrics": {
            "note": "Approximate metrics from training",
            "accuracy": "~0.87",
            "precision": "~0.78",
            "recall": "~0.63",
            "f1_score": "~0.70",
            "roc_auc": "~0.91"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Single customer churn prediction endpoint
    
    What is @app.post("/predict")?
    ------------------------------
    - @app.post: Handles POST requests (used when sending data)
    - "/predict": The URL path for this endpoint
    - response_model=PredictionResponse: Tells FastAPI what this endpoint returns
    
    How does it work?
    -----------------
    1. User sends a POST request to /predict with customer data (JSON)
    2. FastAPI automatically validates the data using CustomerData model
    3. If valid, FastAPI converts JSON to a CustomerData object
    4. This function runs with that object
    5. FastAPI converts the returned dictionary to JSON
    6. The JSON is sent back to the user
    
    Parameters:
    -----------
    customer : CustomerData
        Customer data validated by Pydantic
        FastAPI automatically parses the JSON request body into this object
    
    Returns:
    --------
    PredictionResponse
        Prediction result with churn probability and risk level
    
    Raises:
    -------
    HTTPException
        If there's an error during preprocessing or prediction
    """
    
    try:
        # Step 1: Preprocess the input data
        # Convert customer data to the format the model expects
        input_df = preprocess_input(customer)
        
        # Step 2: Make prediction
        # predict() returns 0 or 1 (no churn or churn)
        prediction = model.predict(input_df)[0]
        
        # Step 3: Get prediction probability
        # predict_proba() returns probabilities for each class [prob_no_churn, prob_churn]
        # We take the second value [1] which is the probability of churning
        churn_probability = model.predict_proba(input_df)[0][1]
        
        # Step 4: Determine risk level
        risk_level = get_risk_level(churn_probability)
        
        # Step 5: Return the response
        # FastAPI automatically converts this to JSON
        return {
            "prediction": int(prediction),  # Convert numpy int to Python int
            "churn_probability": float(churn_probability),  # Convert numpy float to Python float
            "risk_level": risk_level
        }
    
    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise
    
    except Exception as e:
        # Catch any other unexpected errors
        # Return a 500 Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict_churn(batch_request: BatchPredictionRequest):
    """
    Batch prediction endpoint for multiple customers
    
    This endpoint allows predicting churn for multiple customers in a single request.
    This is more efficient than making individual requests for each customer.
    
    Parameters:
    -----------
    batch_request : BatchPredictionRequest
        Contains a list of customer data objects
    
    Returns:
    --------
    BatchPredictionResponse
        Contains predictions for all customers plus summary statistics
    
    Raises:
    -------
    HTTPException
        If there's an error during batch processing
    """
    
    try:
        # Initialize results list
        predictions = []
        churn_count = 0
        
        # Process each customer
        # enumerate() gives us both the index and the customer data
        for idx, customer in enumerate(batch_request.customers):
            try:
                # Preprocess customer data
                input_df = preprocess_input(customer)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                churn_probability = model.predict_proba(input_df)[0][1]
                risk_level = get_risk_level(churn_probability)
                
                # Count churners
                if prediction == 1:
                    churn_count += 1
                
                # Add to results
                predictions.append({
                    "customer_index": idx,  # Which customer in the batch
                    "prediction": int(prediction),
                    "churn_probability": float(churn_probability),
                    "risk_level": risk_level
                })
            
            except Exception as e:
                # If one customer fails, include the error but continue with others
                predictions.append({
                    "customer_index": idx,
                    "error": str(e)
                })
        
        # Return batch results
        return {
            "predictions": predictions,
            "total_customers": len(batch_request.customers),
            "total_churn": churn_count
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
This section only runs when you execute this file directly.
It won't run when FastAPI imports this file.

To run the API:
--------------
In your terminal, navigate to the project directory and run:

    uvicorn main:app --reload

What does this command mean?
- uvicorn: The ASGI server that runs FastAPI applications
- main: The name of this Python file (main.py)
- app: The FastAPI instance we created above
- --reload: Automatically restart the server when code changes (useful for development)

The API will start at: http://127.0.0.1:8000

Accessing the API:
------------------
1. Interactive docs: http://127.0.0.1:8000/docs (Swagger UI)
2. Alternative docs: http://127.0.0.1:8000/redoc (ReDoc)
3. API root: http://127.0.0.1:8000/

These documentation pages are automatically generated by FastAPI!
You can test your API directly from the browser using these interfaces.
"""

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application
    # host="0.0.0.0" makes it accessible from other computers on your network
    # port=8000 is the port number
    # reload=True automatically restarts when you change the code
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
