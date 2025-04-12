import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from utils import get_price_fairness, recommend_similar_cars, preprocess_single_input,calculate_condition_score

# --- Streamlit App ---
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🚗 Car Price Prediction App")

# Load dataset and model
df_raw = pd.read_csv('data/cardekho_dataset.csv')  
final_model = joblib.load('model/car_price_model.pkl')
trained_columns = joblib.load('model/trained_columns.pkl')

st.sidebar.header("🔧 Input Car Details")

brand = st.sidebar.selectbox("Select Car Brand", df_raw['brand'].unique())
model = st.sidebar.selectbox("Select Car Model", df_raw[df_raw['brand'] == brand]['model'].unique())
vehicle_age = st.sidebar.number_input("Vehicle Age", min_value=0, max_value=100, step=1)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.sidebar.selectbox("Transmission Type", ['Manual', 'Automatic'])
owner_type = st.sidebar.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.sidebar.number_input("Mileage (e.g., 18.5 kmpl)", 18.5)
engine = st.sidebar.number_input("Engine (e.g., 1197 CC)", 1197)
max_power = st.sidebar.number_input("Max Power (e.g., 81.80 bhp)", 81.80)
selling_price = st.sidebar.number_input("Selling Price (in ₹)", min_value=0, max_value=1000000000, step=10000)
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual'])

# Create input dict
sample_input = {
    'brand': brand,
    'model': model,
    'vehicle_age': vehicle_age,
    'km_driven': km_driven,
    'fuel_type': fuel_type,
    'transmission_type': transmission_type,
    'owner_type': owner_type,
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'selling_price': selling_price,
    'seller_type': seller_type
}


# Preprocess sample input
processed_input = preprocess_single_input(sample_input, trained_columns)
log_price_pred = final_model.predict(processed_input)[0]
predicted_price = round(np.expm1(log_price_pred), 2)

# Prediction Output
st.subheader("🚗 Predicted Selling Price")
st.success(f"₹{predicted_price:,.2f}")

# Fairness Check
fairness_status = get_price_fairness(predicted_price, sample_input, df_raw)
st.subheader("📊 Price Fairness Status")
st.info(fairness_status)




# Recommendations
predicted_price, recommendations = recommend_similar_cars(
    sample_input, df_raw, final_model, trained_columns
)

st.subheader("🔍 Similar Cars You Can Consider")
if recommendations.empty:
    st.warning("😕 No similar cars found in this price range.")
else:
    # Define the desired columns
    desired_cols = [
        'vehicle_age', 'brand', 'model', 'km_driven',
        'fuel_type_Petrol', 'transmission_type_Manual', 'selling_price'
    ]

    # Select only available columns from recommendations
    available_cols = [col for col in desired_cols if col in recommendations.columns]

    # Display dataframe
    st.dataframe(recommendations[available_cols])


st.markdown("## 🔍 Compare Two Cars Side by Side")

# Select Car 1
st.markdown("### 🚗 Car 1")
car1_brand = st.selectbox("Select Brand for Car 1", df_raw['brand'].unique(), key='car1_brand')
car1_model = st.selectbox("Select Model for Car 1", df_raw[df_raw['brand'] == car1_brand]['model'].unique(), key='car1_model')

# Select Car 2
st.markdown("### 🚙 Car 2")
car2_brand = st.selectbox("Select Brand for Car 2", df_raw['brand'].unique(), key='car2_brand')
car2_model = st.selectbox("Select Model for Car 2", df_raw[df_raw['brand'] == car2_brand]['model'].unique(), key='car2_model')

car1_data = df_raw[(df_raw['brand'] == car1_brand) & (df_raw['model'] == car1_model)].sort_values(by='selling_price', ascending=False).head(1)
car2_data = df_raw[(df_raw['brand'] == car2_brand) & (df_raw['model'] == car2_model)].sort_values(by='selling_price', ascending=False).head(1)

st.markdown("### 📊 Comparison Table")
import pandas as pd

# Ensure both car selections exist
if not car1_data.empty and not car2_data.empty:
    # Create a DataFrame for comparison
    comparison_data = {
        "Specification": [
            "Selling Price (₹)",
            "Fuel Type",
            "Vehicle Age",
            "KM Driven",
            "Estimated Maintenance (₹)"
        ],
        car1_model: [
            car1_data['selling_price'].values[0],
            car1_data['fuel_type'].values[0],
            car1_data['vehicle_age'].values[0],
            car1_data['km_driven'].values[0],
            int(car1_data['vehicle_age'].values[0]) * 10000
        ],
        car2_model: [
            car2_data['selling_price'].values[0],
            car2_data['fuel_type'].values[0],
            car2_data['vehicle_age'].values[0],
            car2_data['km_driven'].values[0],
            int(car2_data['vehicle_age'].values[0]) * 10000
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)
else:
    st.warning("One or both selected cars are not found in the dataset.")


# Input fields for user
st.title("Car Condition Scoring")

# Button to calculate score
if st.button('Calculate Condition Score'):
    score = calculate_condition_score(vehicle_age, km_driven, mileage, engine, max_power)
    
    # Show the calculated score
    st.write(f"Car Condition Score: {score:.2f}/10")
    
    if score >= 8.5:
        st.success("🚗 Excellent condition!")
    elif score >= 7:
        st.info("👍 Good condition.")
    elif score >= 6:
        st.warning("⚠️ Fair condition.")
    else:
        st.error("🔧 Poor condition. Consider a detailed inspection.")
    
# --- Visualizations ---
st.subheader("📈 Insights & Visualizations")

# 1. Distribution of Selling Prices (Similar Cars)
if not recommendations.empty:
    fig, ax = plt.subplots()
    sns.histplot(df_raw[df_raw['model'] == sample_input['model']]['selling_price'], kde=True, ax=ax)
    ax.axvline(predicted_price, color='red', linestyle='--', label='Predicted Price')
    ax.set_title('Selling Price Distribution for Same Model')
    ax.set_xlabel('Selling Price')
    ax.legend()
    st.pyplot(fig)

# 2. KM Driven vs Price (for brand)
fig2, ax2 = plt.subplots()
brand_df = df_raw[df_raw['brand'] == sample_input['brand']]
sns.scatterplot(data=brand_df, x='km_driven', y='selling_price', alpha=0.5, ax=ax2)
ax2.set_title(f"Price vs KM Driven ({sample_input['brand']})")
st.pyplot(fig2)

# 3. Car Count per Brand (Top 10)
st.subheader("🏷️ Most Common Brands in Dataset")
top_brands = df_raw['brand'].value_counts().nlargest(10)
fig3, ax3 = plt.subplots()
sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax3)
ax3.set_xlabel("Number of Cars")
ax3.set_title("Top 10 Brands")
st.pyplot(fig3)

