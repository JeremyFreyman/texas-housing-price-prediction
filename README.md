# Texas Housing Price Prediction

A machine learning project that compares five regression models for predicting residential listing prices in the 2024 Texas housing market. Built as the final project for **CSCI-497 (AI and Data Science)** at Texas A&M University-Commerce.

## Overview

Predicting housing prices is a classic regression problem. We wanted to know whether more complex models would actually beat a simple baseline when the dataset is small (500 listings) and key information like exact location is missing.

**Short answer:** they didn't. Plain linear regression had the lowest error, and the more complex ensemble models performed worse.

## Dataset

- **Source:** Texas Real Estate Trends 2024 (Kaggle), 500 property listings
- **Target:** `listPrice`
- **Features used:** `sqft`, `beds`, `stories`, `year_built`, `type` (property type)
- The dataset is from the post-pandemic Texas market, which has seen big price swings.

## Methods

1. **Preprocessing:** dropped listings with missing values in the target or feature columns, and one-hot encoded the categorical property `type`.
2. **Split:** 80% training / 20% testing (`random_state=42`).
3. **Models compared:**
   - Linear Regression (baseline)
   - Lasso Regression (`alpha=1.0`)
   - Decision Tree Regressor
   - Random Forest Regressor (`n_estimators=100`)
   - Gradient Boosting Regressor
4. **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

## Results

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| **Linear Regression (baseline)** | **$188,991** | **$345,272** | **-0.289** |
| Lasso Regression | $188,987 | $345,270 | -0.289 |
| Decision Tree | $260,745 | $431,335 | -1.012 |
| Random Forest | $229,992 | $448,493 | -1.175 |
| Gradient Boosting | $248,771 | $483,376 | -1.527 |

## Key Takeaways

- **Simpler won.** Linear and Lasso regression had the lowest error. The tree-based models likely overfit the small dataset.
- **All models had negative R²,** meaning none predicted better than simply guessing the average price. Square footage, bedrooms, stories, and year built are not enough to predict price across a market as varied as Texas.
- **Location matters most.** The same square footage is worth far more in downtown Austin or Dallas than in a rural area, and the dataset has no detailed location feature.
- **Data quality and volume matter as much as the algorithm.** For comparison, the Ames Housing study uses 70+ variables and runs into the opposite problem (overfitting from too many features).

## What We'd Do Differently

- Use a much larger dataset (well beyond 500 listings)
- Add location features such as city, ZIP code, or neighborhood
- Tune hyperparameters with cross-validation instead of mostly default settings

## Tools

- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook

## How to Run

1. Download the dataset from Kaggle (Texas Real Estate Trends 2024) and save it as `real_estate_texas_500_2024.csv` in the same folder as the notebook.
2. Install the requirements:
   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```
3. Open the notebook and run all cells:
   ```bash
   jupyter notebook
   ```

## Reference

- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education. https://jse.amstat.org/v19n3/decock.pdf
