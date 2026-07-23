# War Economy Predictor (WEPIS)

## About the Project
This is a machine learning project made by a team of 4 (all 2nd year students) as part of our ML coursework. The idea was to see how wars affect a country's economy, and try to predict changes in things like GDP, stock market performance, and gold prices using historical data from past wars.

We built a small web app called **WEPIS (War Economic Prediction and Impact System)** using Streamlit, where you can select a region, war type, and other details, and it gives you predicted % changes in GDP, stock market, and gold prices based on our trained ML models.

## My Contribution
I was mainly responsible for:
- Collecting and organizing the historical economic data used in this project
- Cleaning and preprocessing the dataset (handling missing values, encoding categorical columns, etc.)
- Training and evaluating the machine learning models used for prediction

This was my first time working on a full ML project from data collection to a working app, and it taught me a lot about how messy real-world data can be and how much preprocessing actually matters before you even get to modeling.

## Features
- Predicts % change in GDP after a war
- Predicts % change in stock market performance
- Predicts % change in gold prices
- Correlation heatmap to visualize relationships between economic indicators
- Clean, custom-styled Streamlit interface (dark theme)

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit
- Joblib (for saving/loading models)

## Project Structure
```
War-Economy-Predictor/
│── app.py
│── requirements.txt
│── war_economics_final.csv
│── model_gdp_pct_change.pkl
│── model_gold_pct_change.pkl
│── model_stock_pct_change.pkl
│── imputer.pkl
│── label_encoder.pkl
│── .gitignore
│── README.md
```

## Installation
Clone the repository:
```bash
git clone https://github.com/prarthyajain-afk/War-Economy-Predictor.git
```
Navigate to the project folder:
```bash
cd War-Economy-Predictor
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run app.py
```

## Dataset
The dataset (`war_economics_final.csv`) contains economic indicators (GDP, inflation, oil prices, gold prices, stock market change, etc.) before and after various wars across different countries and regions. It's a fairly small dataset (~60 records) since historical war-economy data isn't easy to find and put together, so some columns like stock market change have missing values for a good chunk of the entries. We handled this using imputation, but it's a limitation we're aware of and want to improve going forward.

## What I Learned
- How to collect and clean real-world data that isn't in a nice, ready-to-use format
- How to handle missing values properly using imputation instead of just dropping rows
- Training and comparing regression models with Scikit-learn and XGBoost
- Building a working end-to-end ML app with Streamlit
- Working in a team and splitting up an ML project into manageable parts

## Future Improvements
- Add more historical wars/conflicts to grow the dataset
- Try more advanced models to improve prediction accuracy
- Deploy the app online so anyone can try it without setup
- Add more charts/visualizations for better insights
- Include more economic indicators (like unemployment rate, trade balance, etc.)

## Team Members
- Prarthya Jain
- Rachana CU
- Dharani Krishna Sahithi
- PC Suma Krishna

## Author
**Prarthya Jain**
GitHub: https://github.com/prarthyajain-afk

This repo shows my part of the project — mainly data collection, preprocessing, and model training.