# Customer Churn Prediction - EDA & Machine Learning

A comprehensive end-to-end data science project for predicting customer churn using exploratory data analysis and multiple machine learning algorithms.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Analysis Pipeline](#analysis-pipeline)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project performs a complete analysis of customer churn data for a banking institution. It includes:
- **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate analysis
- **Data Cleaning & Preprocessing**: Handling missing values, feature engineering, and encoding
- **Machine Learning Models**: Multiple algorithms with hyperparameter tuning
- **Model Comparison**: Comprehensive evaluation and selection of best model

## ✨ Features

- **Comprehensive EDA**
  - Data quality assessment and cleaning
  - Statistical summaries and distributions
  - Correlation analysis
  - Visual insights with 25+ charts and plots

- **Advanced Machine Learning**
  - Logistic Regression with C parameter optimization
  - Decision Trees (default and tuned)
  - Random Forest ensemble methods
  - Hyperparameter tuning using Grid Search CV
  - Cross-validation for model validation

- **Detailed Model Evaluation**
  - Confusion matrices
  - ROC curves and AUC scores
  - Precision, Recall, F1-Score metrics
  - Feature importance analysis

## 📊 Dataset

The dataset (`churn.csv`) contains 10,000 customer records with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| CreditScore | Customer credit score | Numerical |
| Geography | Customer location (France, Spain, Germany) | Categorical |
| Gender | Customer gender | Categorical |
| Age | Customer age | Numerical |
| Tenure | Years with the bank | Numerical |
| Balance | Account balance | Numerical |
| NumOfProducts | Number of products owned | Numerical |
| HasCrCard | Has credit card (0/1) | Binary |
| IsActiveMember | Active member status (0/1) | Binary |
| EstimatedSalary | Estimated salary | Numerical |
| Exited | Churned (1) or Not (0) | Binary (Target) |

**Note**: RowNumber, CustomerId, and Surname are dropped during preprocessing as they don't contribute to prediction.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone or Download the Repository

```bash
# If using Git
git clone <repository-url>
cd "Project 2 - Customer Churn Prediction"

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version

# Check if all packages are installed
pip list
```

### Step 5: Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Step 6: Open and Run the Notebook

1. In the Jupyter interface, navigate to `customer_churn_eda.ipynb`
2. Click to open the notebook
3. Run cells sequentially using `Shift + Enter` or use "Run All" from the Cell menu
4. Wait for all cells to complete execution (Grid Search may take several minutes)

## 📁 Project Structure

```
Project 2 - Customer Churn Prediction/
│
├── data/
│   └── churn.csv                    # Dataset file
│
├── customer_churn_eda.ipynb         # Main Jupyter notebook
├── eda_analysis.py                  # Python script version (optional)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🔄 Analysis Pipeline

### 1. Data Loading & Cleaning
- Import necessary libraries
- Load dataset from CSV
- Check for missing values and duplicates
- Remove unnecessary columns
- Encode categorical variables

### 2. Exploratory Data Analysis
- **Basic EDA**: Shape, info, statistical summaries
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationships between features and target
- **Multivariate Analysis**: Complex interactions and patterns

### 3. Data Preprocessing for ML
- Label encoding for categorical variables
- Train-test split (80-20) with stratification
- Feature scaling for algorithms that require it

### 4. Model Training & Evaluation
- Train multiple models with default parameters
- Hyperparameter tuning using Grid Search CV
- 10-fold cross-validation
- Comprehensive performance evaluation

### 5. Model Comparison & Selection
- Compare all models across multiple metrics
- Visualize performance differences
- Select best model for deployment

## 🤖 Models Implemented

### 1. Logistic Regression
- **C Parameter Analysis**: Tested 7 different regularization strengths
- **Threshold Tuning**: Analyzed 9 threshold values for precision-recall tradeoff
- Best for interpretability and baseline performance

### 2. Decision Tree
- **Default Model**: No hyperparameter tuning
- **Tuned Model**: Grid Search over 6 parameters
  - max_depth: [3, 5, 7, 10, 15, None]
  - min_samples_split: [2, 5, 10, 20]
  - min_samples_leaf: [1, 2, 4, 8]
  - criterion: ['gini', 'entropy']
  - max_features: ['sqrt', 'log2', None]

### 3. Random Forest
- **Default Model**: 100 trees with default parameters
- **Tuned Model**: Grid Search over 6 parameters
  - n_estimators: [100, 200, 300]
  - max_depth: [5, 10, 15, 20, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - max_features: ['sqrt', 'log2']
  - bootstrap: [True, False]

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.81 | ~0.66 | ~0.52 | ~0.58 | ~0.86 |
| Decision Tree (Default) | ~0.79 | ~0.60 | ~0.52 | ~0.56 | ~0.78 |
| Decision Tree (Tuned) | ~0.86 | ~0.75 | ~0.61 | ~0.67 | ~0.85 |
| Random Forest (Default) | ~0.86 | ~0.76 | ~0.60 | ~0.67 | ~0.89 |
| **Random Forest (Tuned)** | **~0.87** | **~0.78** | **~0.63** | **~0.70** | **~0.91** |

**Winner: Tuned Random Forest** 🏆
- Highest scores across all metrics
- Best generalization with minimal overfitting
- Most robust to variations in data

## 💻 Usage

### Running the Complete Analysis

```python
# Open the notebook in Jupyter
jupyter notebook customer_churn_eda.ipynb

# Run all cells in order (Runtime: 5-10 minutes depending on hardware)
```

### Using Individual Sections

The notebook is organized into clear sections. You can run specific parts:

1. **EDA Only**: Run cells 1-44 for exploratory analysis
2. **ML Models Only**: Run cells 1-7 (setup) and then 45+ for modeling
3. **Specific Model**: Navigate to the model section and run those cells

### Customizing the Analysis

You can modify parameters in the notebook:

```python
# Change train-test split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modify hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],  # Your custom values
    'max_depth': [10, 20, 30]
}

# Adjust cross-validation folds
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # Change from 10 to 5
```

## 🔍 Key Findings

### Business Insights

1. **Age is Critical**: Customers aged 40-60 have significantly higher churn rates
2. **Geography Matters**: German customers show 2x higher churn than France/Spain
3. **Gender Gap**: Female customers have ~10% higher churn rate
4. **Product Paradox**: Customers with 3-4 products have extremely high churn (80%+)
5. **Activity Impact**: Inactive members are 2x more likely to churn
6. **Balance Effect**: Customers with higher balances show slight increase in churn

### Technical Insights

1. **Ensemble Methods Win**: Random Forest outperformed single models
2. **Tuning is Essential**: Hyperparameter optimization improved all models by 5-10%
3. **Cross-Validation Critical**: Helped identify models with better generalization
4. **Feature Importance**: Age, Balance, and Geography are top predictors
5. **Class Imbalance**: 20% churn rate requires careful threshold tuning

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and tools
- **Jupyter Notebook**: Interactive development environment

## 📝 Notes

- Grid Search hyperparameter tuning may take 5-15 minutes depending on your hardware
- The notebook includes ~50+ visualizations; rendering may take time
- For production deployment, consider saving the trained model using `joblib` or `pickle`
- Results may vary slightly due to random state in train-test split

## 🤝 Contributing

Contributions are welcome! Some ideas:

- Add more advanced models (XGBoost, LightGBM, Neural Networks)
- Implement SMOTE for handling class imbalance
- Add feature engineering techniques
- Create a web interface using Streamlit or Flask
- Deploy the model to cloud platforms

## 📄 License

This project is available for educational and personal use.

## 👤 Author

Created as part of an AI Engineering course project.

## 🙏 Acknowledgments

- Dataset sourced from banking customer records
- Inspired by real-world churn prediction challenges
- Built using open-source libraries and tools

---

**Last Updated**: December 2025

**Status**: ✅ Complete

For questions or suggestions, please open an issue or contact the repository owner.
