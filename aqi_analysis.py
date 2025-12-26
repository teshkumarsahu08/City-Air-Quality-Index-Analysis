# 	" City Air Quality Index Analysis 
# “First, I created a virtual environment and installed required libraries.
# Then I loaded air quality data from Hugging Face using pandas.
# I cleaned the data, extracted month-wise information, and analyzed PM2.5 trends.
# I visualized monthly mean and variance using matplotlib and tracked everything using MLflow.”
# 
# 
#============================
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. Load Hugging Face Dataset
# ===============================
df = pd.read_csv(
    "hf://datasets/hmnshudhmn24/global-air-quality-major-cities/global_air_quality_data_10000.csv"
)

print("Dataset Columns:")
print(df.columns)

# ===============================
# 2. Select Required Columns
# ===============================
df = df[['City', 'Date', 'PM2.5']]

# Rename for ease
df.rename(columns={'PM2.5': 'PM25'}, inplace=True)

# ===============================
# 3. Data Cleaning
# ===============================
df.dropna(inplace=True)

df['PM25'] = pd.to_numeric(df['PM25'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df.dropna(inplace=True)

# ===============================
# 4. Feature Engineering
# ===============================
df['Month'] = df['Date'].dt.month


# ===============================
# 5. Pick a Valid City Automatically
# ===============================
city_name = df['City'].value_counts().idxmax()
print(f"Using city: {city_name}")

city_df = df[df['City'] == city_name]

if city_df.empty:
    raise ValueError("City dataset is empty. Cannot continue.")

# ===============================
# 6. Monthly Mean AQI Trend
# ===============================
monthly_avg = city_df.groupby('Month')['PM25'].mean()

plt.figure()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.xlabel("Month")
plt.ylabel("Average PM2.5")
plt.title(f"Monthly Air Quality Trend in {city_name}")
plt.grid(True)
plt.savefig("monthly_air_quality_trend.png")
plt.close()

# ===============================
# 7. Monthly Variance Analysis (SAFE)
# ===============================
monthly_variance = city_df.groupby('Month')['PM25'].var().dropna()

plt.figure()
plt.bar(monthly_variance.index, monthly_variance.values)
plt.xlabel("Month")
plt.ylabel("PM2.5 Variance")
plt.title(f"Monthly Air Quality Variance in {city_name}")
plt.grid(True)
plt.savefig("monthly_air_quality_variance.png")
plt.close()

# ===============================
# 8. MLflow Tracking
# ===============================
mlflow.set_experiment("City Air Quality Analysis")

with mlflow.start_run():
    mlflow.log_param("City", city_name)
    mlflow.log_metric("Mean_PM25", city_df['PM25'].mean())
    mlflow.log_metric("PM25_Variance", city_df['PM25'].var())
    mlflow.log_artifact("monthly_air_quality_trend.png")
    mlflow.log_artifact("monthly_air_quality_variance.png")

print("✅ Project executed successfully!")
print("👉 Run: mlflow ui")
