# House Price Prediction

A machine learning project that predicts residential house prices based on structural and spatial features using a Linear Regression model. This repository was developed as part of the Prodigy InfoTech Machine Learning Internship (Task 01).

## 🚀 Overview

This project implements a linear regression model to estimate the `SalePrice` of houses. The model is trained on specific features extracted from the dataset, evaluating physical attributes like living area space, room counts, and lot size to generate predictive pricing.

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas**: Data manipulation and CSV parsing.
* **Scikit-Learn**: Machine learning model training, data splitting (`train_test_split`), and evaluation metrics.

## 📊 Dataset Requirements

The script expects two dataset files located in the `/content/` directory (modify the paths in the script if running locally outside of Google Colab):

* `train.csv`: Training data containing the features and the target variable.
* `test.csv`: Testing data used for generating the final submission file.

**Selected Features:**

* `GrLivArea`: Above grade (ground) living area square feet
* `BedroomAbvGr`: Number of bedrooms above basement level
* `FullBath`: Full bathrooms above grade
* `LotArea`: Lot size in square feet

## ⚙️ Installation & Execution

1. **Clone the repository:**
```bash
git clone https://github.com/Thambajagadeesh/house-price-prediction.git

```


2. **Install dependencies:**
```bash
pip install pandas scikit-learn

```


3. **Run the script:**
```bash
python model.py

```



## 📈 Outputs & Metrics

During execution, the script splits the training data into an 80/20 train-validation set and outputs the following evaluation metrics to the console:

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)
* **R² Score** (Coefficient of Determination)

Final predictions for the `test.csv` dataset are automatically exported to a new file named `house_price_predictions.csv`, formatted with the `Id` and `PredictedPrice` columns for easy submission or review.

---


