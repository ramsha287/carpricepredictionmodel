
import pandas as pd
import numpy as np


def preprocess_single_input(input_dict, trained_columns):
    df_input = pd.DataFrame([input_dict])
    for col in ['mileage', 'engine', 'max_power']:
        df_input[col] = df_input[col].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float)
    df_input['log_km'] = np.log1p(df_input['km_driven'])
    df_input['log_engine'] = np.log1p(df_input['engine'])
    df_input['power_to_engine'] = df_input['max_power'] / (df_input['engine'] + 1)
    df_input['price_per_km'] = df_input['selling_price'] / (df_input['km_driven'] + 1)
    cat_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
    df_input = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)
    df_input = df_input.reindex(columns=trained_columns, fill_value=0)
    return df_input

def get_price_fairness(predicted_price, input_dict, df_raw):
    brand = input_dict['brand']
    model = input_dict['model']
    vehicle_age = input_dict['vehicle_age']


    similar_cars = df_raw[
        (df_raw['brand'].str.lower() == brand.lower()) &
        (df_raw['model'].str.lower() == model.lower()) &
        (df_raw['vehicle_age'].astype(str) == str(vehicle_age))
    ]


    avg_price = similar_cars['selling_price'].mean()
    if np.isnan(avg_price):
        return "No Comparison"
    elif abs(predicted_price - avg_price) <= 0.1 * avg_price:
        return "✅ Fair Deal"
    elif predicted_price > avg_price:
        return "⚠️ Overpriced"
    else:
        return "💰 Underpriced"

def recommend_similar_cars(input_dict, original_df, model, trained_columns, top_n=5, price_tolerance=0.15):
    processed_input = preprocess_single_input(input_dict, trained_columns)
    predicted_log_price = model.predict(processed_input)[0]
    predicted_price = np.expm1(predicted_log_price)

    lower_price = predicted_price * (1 - price_tolerance)
    upper_price = predicted_price * (1 + price_tolerance)

    # Filter by price range
    price_filtered = original_df[
        (original_df['selling_price'] >= lower_price) &
        (original_df['selling_price'] <= upper_price)
    ]

    # Build masks safely
    fuel_col = 'fuel_type_' + input_dict['fuel_type']
    transmission_col = 'transmission_type_' + input_dict['transmission_type']

    fuel_mask = (price_filtered[fuel_col] == 1) if fuel_col in price_filtered.columns else True
    transmission_mask = (price_filtered[transmission_col] == 1) if transmission_col in price_filtered.columns else True
    brand_mask = price_filtered['brand'] == input_dict['brand']
    model_mask = price_filtered['model'] == input_dict['model']

    # Apply all filters
    feature_filtered = price_filtered[brand_mask & model_mask & fuel_mask & transmission_mask]

    if feature_filtered.empty:
        return predicted_price, pd.DataFrame()

    recommendations = feature_filtered.copy()
    recommendations['selling_price'] = recommendations['selling_price'].apply(lambda x: f"₹{int(x):,}")
    recommendations = recommendations.rename(columns={'vehicle_age': 'age (years)'})
    top_recommendations = recommendations.sort_values(by='km_driven').head(top_n)

    return predicted_price, top_recommendations

# Function to calculate car condition score
def calculate_condition_score(vehicle_age, km_driven, mileage, engine, max_power):
    # Define weightage for each factor (you can adjust based on dataset insights or domain knowledge)
    age_weight = 0.25
    mileage_weight = 0.2
    engine_weight = 0.15
    power_weight = 0.1

    # Score based on vehicle age (lower age means better condition)
    if vehicle_age <= 3:
        age_score = 10
    elif vehicle_age <= 5:
        age_score = 8
    elif vehicle_age <= 8:
        age_score = 6
    else:
        age_score = 4

    # Score based on mileage (lower mileage means better condition)
    if km_driven < 20000:
        mileage_score = 10
    elif km_driven < 50000:
        mileage_score = 8
    elif km_driven < 100000:
        mileage_score = 6
    else:
        mileage_score = 4

    # Score based on mileage (higher mileage per liter is better)
    if mileage >= 15:
        mileage_efficiency_score = 10
    elif mileage >= 12:
        mileage_efficiency_score = 8
    else:
        mileage_efficiency_score = 6

    # Score based on engine capacity (higher engine capacity might indicate better performance)
    if engine >= 2000:
        engine_score = 10
    elif engine >= 1500:
        engine_score = 8
    else:
        engine_score = 6

    # Score based on max power (more power may indicate better performance)
    if max_power >= 150:
        power_score = 10
    elif max_power >= 100:
        power_score = 8
    else:
        power_score = 6

    # Calculate total score
    total_score = (age_score * age_weight + 
                   mileage_score * mileage_weight + 
                   mileage_efficiency_score * mileage_weight + 
                   engine_score * engine_weight + 
                   power_score * power_weight)

    return total_score