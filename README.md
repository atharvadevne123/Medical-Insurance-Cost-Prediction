# 🏥 Medical Insurance Cost Prediction

This project predicts medical insurance costs based on a person's demographic and health-related information using machine learning techniques. The goal is to help insurance providers estimate charges more accurately and transparently.

---

## 📊 Dataset

- **Source**: `insurance.csv`
- **Features**:
  - `age`: Age of the primary beneficiary
  - `sex`: Gender
  - `bmi`: Body Mass Index
  - `children`: Number of dependents
  - `smoker`: Smoking status
  - `region`: Residential region in the U.S.
  - `charges`: Target variable (medical insurance cost)

---

## 🧠 ML Workflow Summary

- **Data Preprocessing**:
  - Handled categorical encoding (`sex`, `region`, `smoker`)
  - Checked for missing values and outliers
  - Scaled/normalized features where required

- **Exploratory Data Analysis (EDA)**:
  - Visualized distributions and correlations using Seaborn/Matplotlib
  - Analyzed relationships between `charges` and other variables

- **Modeling Techniques**:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - (Optional: Grid Search for Hyperparameter Tuning)

- **Model Evaluation**:
  - Metrics used: MAE, MSE, RMSE, R² Score
  - Compared model performances to select the best fit

---

## 🛠️ Tools & Libraries

- Python (Jupyter Notebook)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LinearRegression, RandomForest, etc.)

---

## 📈 Results

- The best-performing model (e.g., Random Forest) accurately predicted insurance costs with a high R² score.
- Key predictors: **smoker status**, **age**, and **BMI** had the strongest influence on `charges`.

---

## 📁 Project Structure

```
├── insurance.csv                      # Input dataset
├── Medical Insurance Cost Prediction.ipynb  # Jupyter notebook with EDA and ML pipeline
├── README.md                         # Project overview and documentation
```
---

## 🧪 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/insurance-cost-prediction.git
   cd insurance-cost-prediction
   ```
2.	Install required packages:


    ```
    pip install -r requirements.txt
    ```

3.	Launch the notebook:

```
jupyter notebook Medical\ Insurance\ Cost\ Prediction.ipynb
```

---

## ✍️ Author

## Atharva Devne ##
