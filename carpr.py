import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.write("Predict Car Price using Machine Learning")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():

    df = pd.read_csv("carprice.csv")

    # Convert price to numeric
    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    fuel_encoder = LabelEncoder()
    location_encoder = LabelEncoder()
    type_encoder = LabelEncoder()

    df["fuel-type"] = fuel_encoder.fit_transform(
        df["fuel-type"]
    )

    df["engine-location"] = location_encoder.fit_transform(
        df["engine-location"]
    )

    df["engine-type"] = type_encoder.fit_transform(
        df["engine-type"]
    )

    return (
        df,
        fuel_encoder,
        location_encoder,
        type_encoder
    )

df, fuel_encoder, location_encoder, type_encoder = load_data()

# =====================================
# FEATURES
# =====================================
X = df.drop("price", axis=1)
y = df["price"]

# =====================================
# MODEL
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

# =====================================
# METRICS
# =====================================
c1, c2, c3 = st.columns(3)

c1.metric("📊 Records", len(df))
c2.metric("🎯 Features", X.shape[1])
c3.metric("📈 Model Score", round(score, 2))

st.divider()

# =====================================
# USER INPUT
# =====================================
st.subheader("🚘 Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    fuel_type = st.selectbox(
        "Fuel Type",
        ["gas", "diesel"]
    )

    engine_location = st.selectbox(
        "Engine Location",
        ["front", "rear"]
    )

    horsepower = st.slider(
        "Horsepower",
        50,
        300,
        100
    )

    peak_rpm = st.slider(
        "Peak RPM",
        4000,
        7000,
        5000
    )

with col2:

    engine_type = st.selectbox(
        "Engine Type",
        ["dohc", "ohc", "ohcv", "ohcf"]
    )

    city_mpg = st.slider(
        "City MPG",
        10,
        60,
        25
    )

    highway_mpg = st.slider(
        "Highway MPG",
        10,
        70,
        30
    )

# =====================================
# PREDICTION
# =====================================
if st.button("🚀 Predict Price"):

    input_df = pd.DataFrame({

        "fuel-type": [
            fuel_encoder.transform(
                [fuel_type]
            )[0]
        ],

        "engine-location": [
            location_encoder.transform(
                [engine_location]
            )[0]
        ],

        "engine-type": [
            type_encoder.transform(
                [engine_type]
            )[0]
        ],

        "horsepower": [horsepower],

        "peak-rpm": [peak_rpm],

        "city-mpg": [city_mpg],

        "highway-mpg": [highway_mpg]
    })

    predicted_price = model.predict(
        input_df
    )[0]

    st.success(
        f"💰 Predicted Car Price: ${predicted_price:,.2f}"
    )

# # =====================================
# # FEATURE IMPORTANCE
# # =====================================
# st.subheader("📊 Feature Importance")

# importance = model.feature_importances_

# fig, ax = plt.subplots(figsize=(8,4))

# ax.bar(
#     X.columns,
#     importance
# )

# plt.xticks(rotation=45)

# st.pyplot(fig)

# # =====================================
# # DATASET PREVIEW
# # =====================================
# st.subheader("📄 Dataset Preview")

# st.dataframe(df.head())

# # =====================================
# # STATISTICS
# # =====================================
# st.subheader("📈 Dataset Statistics")

# st.dataframe(df.describe())

# # =====================================
# # FOOTER
# # =====================================
# st.markdown("---")
# st.markdown(
#     "### Developed using Streamlit + Random Forest Regression"
# )