# FastAPI Setup and Usage Guide

## 📚 Complete FastAPI Tutorial - From Basics to Deployment

---

## 🎯 Step-by-Step Setup Instructions

### **Step 1: Train and Save the Model**

First, we need to train our best model and save it for the API to use.

```bash
# Make sure you're in the project directory
cd "Project 2 - Customer Churn Prediction"

# Run the training script
python train_and_save_model.py
```

**What this does:**
- Loads the churn dataset
- Trains the Random Forest model with hyperparameter tuning
- Saves the trained model to `model/best_churn_model.pkl`
- Saves the encoders for Geography and Gender
- Creates necessary files for the API

**Expected output:**
```
TRAINING AND SAVING THE BEST MODEL - Random Forest (Tuned)
1. Loading dataset...
   Dataset loaded: 10000 rows, 14 columns
2. Data cleaning...
   ...
✓ Model saved: model/best_churn_model.pkl
```

---

### **Step 2: Install FastAPI Dependencies**

```bash
# Install FastAPI and related packages
pip install -r requirements_api.txt
```

**What gets installed:**
- `fastapi`: The web framework
- `uvicorn`: ASGI server to run the app
- `pydantic`: Data validation library
- Other supporting libraries

---

### **Step 3: Start the FastAPI Server**

```bash
# Run the API server
uvicorn main:app --reload
```

**Command breakdown:**
- `uvicorn`: The server that runs FastAPI
- `main`: The Python file name (main.py)
- `app`: The FastAPI instance in main.py
- `--reload`: Auto-restart when code changes (development mode)

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### **Step 4: Access the API**

Open your browser and visit these URLs:

1. **API Root**: http://127.0.0.1:8000/
2. **Interactive Docs (Swagger)**: http://127.0.0.1:8000/docs
3. **Alternative Docs (ReDoc)**: http://127.0.0.1:8000/redoc

**The `/docs` page is AMAZING!** You can:
- See all available endpoints
- Test the API directly from your browser
- See request/response examples
- View data schemas

---

## 🔍 Understanding FastAPI Components

### **1. What is FastAPI?**

FastAPI is a modern Python web framework for building APIs. Think of it as a way to make your Python functions accessible over the internet.

**Key Features:**
```python
# Traditional Python function
def predict_churn(customer_data):
    # Make prediction
    return prediction

# With FastAPI - Now accessible via HTTP!
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Same function, but now:
    # - Accessible at http://your-api.com/predict
    # - Automatic data validation
    # - Automatic documentation
    return prediction
```

---

### **2. What is Pydantic and BaseModel?**

Pydantic handles data validation automatically.

**Without Pydantic (Manual Validation):**
```python
def predict(data):
    # Manual checks
    if 'CreditScore' not in data:
        raise ValueError("Missing CreditScore")
    if not isinstance(data['CreditScore'], int):
        raise ValueError("CreditScore must be integer")
    if data['CreditScore'] < 300 or data['CreditScore'] > 850:
        raise ValueError("CreditScore must be 300-850")
    # ... repeat for 10 fields! 😰
```

**With Pydantic (Automatic):**
```python
class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    # That's it! Validation is automatic ✨
```

**BaseModel gives you:**
- ✅ Automatic type checking
- ✅ Automatic range validation
- ✅ Clear error messages
- ✅ JSON schema generation
- ✅ API documentation

---

### **3. What does `app` do?**

The `app` object is your API server.

```python
app = FastAPI(title="My API")

# app handles:
# 1. Routing - directing requests to the right function
# 2. Request parsing - converting JSON to Python objects
# 3. Response formatting - converting Python objects to JSON
# 4. Error handling - catching and formatting errors
# 5. Documentation generation - creating /docs automatically
```

---

### **4. Decorators (@app.get, @app.post)**

Decorators tell FastAPI which function handles which URL.

```python
@app.get("/")        # Handles: GET request to http://api.com/
def home():
    return {"message": "Welcome"}

@app.post("/predict")  # Handles: POST request to http://api.com/predict
def predict(data: CustomerData):
    return {"prediction": 1}
```

**HTTP Methods:**
- `GET`: Retrieve data (like viewing a webpage)
- `POST`: Send data (like submitting a form)
- `PUT`: Update data
- `DELETE`: Remove data

---

### **5. Request Flow**

Here's what happens when someone uses your API:

```
1. User sends HTTP request
   POST http://127.0.0.1:8000/predict
   Body: {"CreditScore": 700, "Geography": "France", ...}
   
2. FastAPI receives request
   - Parses JSON
   - Validates against CustomerData model
   - If invalid: Returns 422 error with details
   
3. Calls your function
   predict_churn(customer=CustomerData(...))
   
4. Your function runs
   - Preprocesses data
   - Makes prediction
   - Returns result
   
5. FastAPI formats response
   - Converts Python dict to JSON
   - Adds HTTP headers
   - Sends back to user
   
6. User receives response
   {"prediction": 1, "churn_probability": 0.75, ...}
```

---

## 📡 API Endpoints

### **1. Root Endpoint**

```http
GET http://127.0.0.1:8000/
```

**Response:**
```json
{
  "message": "Customer Churn Prediction API",
  "status": "active",
  "model": "Random Forest (Tuned)",
  "endpoints": {...}
}
```

---

### **2. Health Check**

```http
GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### **3. Model Information**

```http
GET http://127.0.0.1:8000/model-info
```

**Response:**
```json
{
  "model_type": "Random Forest Classifier",
  "features": ["CreditScore", "Geography", ...],
  "geography_options": ["France", "Germany", "Spain"],
  "performance_metrics": {...}
}
```

---

### **4. Single Prediction**

```http
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "CreditScore": 700,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 75000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 100000.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "churn_probability": 0.23,
  "risk_level": "Low"
}
```

---

### **5. Batch Prediction**

```http
POST http://127.0.0.1:8000/batch-predict
Content-Type: application/json

{
  "customers": [
    {
      "CreditScore": 700,
      "Geography": "France",
      ...
    },
    {
      "CreditScore": 400,
      "Geography": "Germany",
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_index": 0,
      "prediction": 0,
      "churn_probability": 0.23,
      "risk_level": "Low"
    },
    {
      "customer_index": 1,
      "prediction": 1,
      "churn_probability": 0.78,
      "risk_level": "High"
    }
  ],
  "total_customers": 2,
  "total_churn": 1
}
```

---

## 🧪 Testing the API

### **Method 1: Using the Browser (Interactive Docs)**

1. Start the API: `uvicorn main:app --reload`
2. Go to: http://127.0.0.1:8000/docs
3. Click on any endpoint (e.g., `/predict`)
4. Click "Try it out"
5. Fill in the example data
6. Click "Execute"
7. See the response!

**This is the EASIEST way to test!**

---

### **Method 2: Using Python Script**

```bash
# Run the test script
python test_api.py
```

This will test all endpoints automatically.

---

### **Method 3: Using curl (Command Line)**

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 700,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0
  }'
```

---

### **Method 4: Using Python requests**

```python
import requests

# Single prediction
url = "http://127.0.0.1:8000/predict"
data = {
    "CreditScore": 700,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🎨 How the API Works Internally

### **Data Validation Flow:**

```python
# 1. User sends JSON
{
  "CreditScore": 700,
  "Age": 35,
  ...
}

# 2. FastAPI validates using CustomerData model
class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850)
    Age: int = Field(..., ge=18, le=100)
    ...

# 3. If valid, creates Python object
customer = CustomerData(
    CreditScore=700,
    Age=35,
    ...
)

# 4. Passes to your function
predict_churn(customer)
```

---

### **Prediction Flow:**

```python
# 1. Preprocess input
input_df = preprocess_input(customer)
# - Validates Geography and Gender
# - Encodes categorical variables to numbers
# - Creates DataFrame with correct feature order

# 2. Make prediction
prediction = model.predict(input_df)[0]
# Returns: 0 or 1

# 3. Get probability
probability = model.predict_proba(input_df)[0][1]
# Returns: 0.0 to 1.0

# 4. Determine risk level
risk = get_risk_level(probability)
# Returns: "Low", "Medium", or "High"

# 5. Return formatted response
return {
    "prediction": int(prediction),
    "churn_probability": float(probability),
    "risk_level": risk
}
```

---

## 🚀 Production Deployment Tips

### **1. Use Production Server**

```bash
# Development (current)
uvicorn main:app --reload

# Production (multiple workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

### **2. Environment Variables**

```python
import os

# Don't hardcode paths
MODEL_PATH = os.getenv("MODEL_PATH", "model")
```

---

### **3. Add CORS (for web apps)**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### **4. Add Logging**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
def predict_churn(customer: CustomerData):
    logger.info(f"Prediction request received")
    # ... rest of code
```

---

## 🐛 Troubleshooting

### **Problem: "Module not found"**
```bash
# Solution: Install dependencies
pip install -r requirements_api.txt
```

---

### **Problem: "Model file not found"**
```bash
# Solution: Train and save model first
python train_and_save_model.py
```

---

### **Problem: "Port already in use"**
```bash
# Solution: Use different port
uvicorn main:app --reload --port 8001
```

---

### **Problem: "422 Validation Error"**
```
This means your input data is invalid.
Check the error message for details:
- Missing required fields?
- Wrong data types?
- Values out of range?
```

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Uvicorn Documentation**: https://www.uvicorn.org/

---

## 🎉 Summary

**You now have:**
1. ✅ Trained ML model saved to disk
2. ✅ FastAPI application with detailed comments
3. ✅ Automatic data validation with Pydantic
4. ✅ Interactive API documentation
5. ✅ Test scripts to verify functionality
6. ✅ Complete understanding of how it works!

**To start using:**
```bash
# 1. Train model (one time)
python train_and_save_model.py

# 2. Start API
uvicorn main:app --reload

# 3. Test it
python test_api.py

# 4. Open docs
# Browser: http://127.0.0.1:8000/docs
```

Enjoy your API! 🚀
