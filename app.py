import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="🌏 Global Climate Agriculture Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

@st.cache_resource
def load_all():
    df = pd.read_csv("climate_change_impact_on_agriculture_2024.csv")
    try:
        model = joblib.load("climate_linear_model.joblib")
    except Exception:
        model = None
    try:
        scaler = joblib.load("climate_scaler.joblib")
    except Exception:
        scaler = None
    return df, model, scaler

df, model, scaler = load_all()

TEMP_COL = detect_column(df, ["Average_Temperature", "Avg_Temperature", "Temperature", "Temperature_C", "Temp", "Mean_Temperature"])
RAIN_COL = detect_column(df, ["Annual_Rainfall_mm", "Annual_Rainfall", "Rainfall_mm", "Rain_mm", "Rainfall"])
YEAR_COL = detect_column(df, ["Year", "year"])
COUNTRY_COL = detect_column(df, ["Country", "country", "Country_Name", "CountryName"])
YIELD_COL = detect_column(df, ["Crop_Yield_MT_per_HA", "Crop_Yield", "Yield_MT_per_HA", "Yield", "Crop_Yield_MT_per_Hectare"])

required_some = []
if YIELD_COL is None:
    required_some.append("crop yield (e.g. 'Crop_Yield_MT_per_HA')")
if YEAR_COL is None:
    required_some.append("year (e.g. 'Year')")
if COUNTRY_COL is None:
    # country is optional for some features, but inform user
    pass

if required_some:
    st.error("The dataset is missing required column(s): " + ", ".join(required_some))
    st.write("Detected columns in your CSV:")
    st.write(list(df.columns))
    st.stop()

if model is None or scaler is None:
    st.error("Saved model or scaler not found or failed to load. Make sure 'climate_linear_model.joblib' and 'climate_scaler.joblib' exist.")
    st.write("Model loaded:", model is not None, "Scaler loaded:", scaler is not None)
    st.stop()

st.title("🌎 Climate Change Impact on Agriculture Dashboard")
st.markdown("<h5 style='color:#00bcd4;'>Real-time global insights, ML forecasting & scenario simulation.</h5>", unsafe_allow_html=True)

st.sidebar.header("🛠️ Navigation")
page = st.sidebar.radio("Select Section", ["🏠 Dashboard", "🗺️ Maps", "🔮 Projections", "📝 Scenario Simulator"])

if COUNTRY_COL in df.columns:
    country_filter = st.sidebar.multiselect("Filter by Country",
                                            options=sorted(df[COUNTRY_COL].dropna().unique()),
                                            default=[])
else:
    country_filter = []

if YEAR_COL in df.columns:
    year_filter = st.sidebar.slider("Filter by Year",
                                    int(df[YEAR_COL].min()),
                                    int(df[YEAR_COL].max()),
                                    int(df[YEAR_COL].min()))
else:
    year_filter = None

filtered_df = df.copy()
if country_filter:
    if COUNTRY_COL in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[COUNTRY_COL].isin(country_filter)]
if year_filter is not None and YEAR_COL in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[YEAR_COL] == year_filter]

if page == "🏠 Dashboard":
    st.header("📊 Global Agriculture Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_yield = filtered_df.get(YIELD_COL, pd.Series([0])).mean()
        st.metric("Average Yield (MT/HA)", f"{avg_yield:.2f}")
    with col2:
        avg_temp = filtered_df.get(TEMP_COL, pd.Series([0])).mean() if TEMP_COL else 0
        st.metric("Average Temperature (°C)", f"{avg_temp:.2f}")
    with col3:
        avg_rain = filtered_df.get(RAIN_COL, pd.Series([0])).mean() if RAIN_COL else 0
        st.metric("Average Rainfall (mm)", f"{avg_rain:.2f}")

    if YEAR_COL in filtered_df.columns and YIELD_COL in filtered_df.columns and COUNTRY_COL in filtered_df.columns:
        fig = px.line(filtered_df, x=YEAR_COL, y=YIELD_COL, color=COUNTRY_COL,
                      title="📈 Crop Yield Trend Over Years", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough columns to draw trend (need Year, Crop Yield and Country).")

elif page == "🗺️ Maps":
    st.header("🗺️ Global Maps")
    if filtered_df.empty:
        st.info("No data to show on map (after filters).")
    elif COUNTRY_COL not in filtered_df.columns:
        st.info("Country column not found — cannot render maps.")
    else:
        agg_dict = {YIELD_COL: "mean"} if YIELD_COL in filtered_df.columns else {}
        if TEMP_COL:
            agg_dict[TEMP_COL] = "mean"
        if not agg_dict:
            st.info("Not enough numeric columns to map (need crop yield or temperature).")
        else:
            grouped = filtered_df.groupby(COUNTRY_COL).agg(agg_dict).reset_index()
            if YIELD_COL in grouped.columns:
                fig_yield = px.choropleth(grouped, locations=COUNTRY_COL, locationmode="country names",
                                          color=YIELD_COL, hover_name=COUNTRY_COL,
                                          title="🌾 Average Crop Yield by Country",
                                          color_continuous_scale="Greens")
                st.plotly_chart(fig_yield, use_container_width=True)
            if TEMP_COL and TEMP_COL in grouped.columns:
                fig_temp = px.choropleth(grouped, locations=COUNTRY_COL, locationmode="country names",
                                         color=TEMP_COL, hover_name=COUNTRY_COL,
                                         title="🌡️ Average Temperature by Country",
                                         color_continuous_scale="Reds")
                st.plotly_chart(fig_temp, use_container_width=True)

elif page == "🔮 Projections":
    st.header("🔮 Future Yield Forecast")
    if YEAR_COL in df.columns and YIELD_COL in df.columns:
        future_years = st.slider("Forecast Up To", int(df[YEAR_COL].max())+1, int(df[YEAR_COL].max())+15, int(df[YEAR_COL].max())+5)
        X = df[[YEAR_COL]].dropna().astype(int)
        y = df[[YIELD_COL]].loc[X.index]
        from sklearn.linear_model import LinearRegression
        trend_model = LinearRegression().fit(X, y.values.ravel())
        future_range = np.arange(int(df[YEAR_COL].max())+1, int(future_years)+1).reshape(-1, 1)
        future_preds = trend_model.predict(future_range)
        df_future = pd.DataFrame({YEAR_COL: future_range.flatten(), "Predicted_Yield": future_preds})
        df_hist = pd.DataFrame({YEAR_COL: df[YEAR_COL], "Predicted_Yield": df[YIELD_COL]})
        final_df = pd.concat([df_hist, df_future]).sort_values(by=YEAR_COL)
        fig = px.line(final_df, x=YEAR_COL, y="Predicted_Yield", title="📈 Historical + Predicted Crop Yield")
        st.plotly_chart(fig, use_container_width=True)
        
        # New addition: Display predicted yields in a table
        st.subheader("📊 Predicted Yields for Future Years")
        st.dataframe(df_future.rename(columns={"Predicted_Yield": "Yield (MT/HA)"}).style.format({"Yield (MT/HA)": "{:.2f}"}))
    else:
        st.info("Need both Year and Crop Yield columns to create projections.")

elif page == "📝 Scenario Simulator":
    st.header("📝 Climate Scenario Simulator")
    st.markdown("Adjust climate variables to simulate crop yield impact.")
    with st.form("sim_form"):
        temp_val = st.number_input("Average Temperature (°C)", value=float(df[TEMP_COL].mean()) if TEMP_COL else 0.0)
        rain_val = st.number_input("Annual Rainfall (mm)", value=float(df[RAIN_COL].mean()) if RAIN_COL else 0.0)
        year_val = st.number_input("Year", value=int(df[YEAR_COL].mean()) if YEAR_COL else 0)
        submitted = st.form_submit_button("Simulate 🌱")

    if submitted:
        user_df = pd.DataFrame([{TEMP_COL: temp_val}]) if TEMP_COL else pd.DataFrame([{}])
        if RAIN_COL:
            user_df[RAIN_COL] = rain_val
        if YEAR_COL:
            user_df[YEAR_COL] = year_val

        df_X = df.copy()
        if YIELD_COL in df_X.columns:
            df_X = df_X.drop(columns=[YIELD_COL], errors=True)

        df_encoded = pd.get_dummies(df_X, drop_first=True)
        user_encoded = pd.get_dummies(user_df, drop_first=True)
        user_encoded = user_encoded.reindex(columns=df_encoded.columns, fill_value=0)

        try:
            X_scaled = scaler.transform(user_encoded)
        except Exception as e:
            st.error("Scaler transform failed. Possible feature-shape mismatch.")
            st.write("Details:", str(e))
            st.write("Model expects features:", getattr(model, "n_features_in_", "unknown"))
            st.write("Available columns used for training/re-indexing:", df_encoded.columns.tolist())
            st.stop()

        try:
            prediction = model.predict(X_scaled)[0]
        except Exception as e:
            st.error("Model prediction failed. See details below.")
            st.write("Details:", str(e))
            if hasattr(model, "coef_"):
                st.write("Model coefficients sample:", model.coef_[:10])
            st.stop()

        st.success(f"🌾 Predicted Yield: **{prediction:.2f} MT/HA**")
        if YIELD_COL in df.columns:
            avg_yield = df[YIELD_COL].mean()
        else:
            avg_yield = None
        bars_x = ["Scenario Yield"] if avg_yield is None else ["Average Yield", "Scenario Yield"]
        bars_y = [prediction] if avg_yield is None else [avg_yield, prediction]
        fig = px.bar(x=bars_x, y=bars_y, title="Scenario Comparison", labels={"x": "", "y": "Yield (MT/HA)"})
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("<h6 style='color:gray;'>Powered by real climate data + ML</h6>", unsafe_allow_html=True)
