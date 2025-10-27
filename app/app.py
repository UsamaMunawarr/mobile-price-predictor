# app.py
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Load Data & Models
# ---------------------------
@st.cache_data(ttl=3600)
def load_data(path="app/mobiles_prices_prediction.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

df = load_data()

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="📱 Mobile Price Predictor", layout="wide")

page = st.sidebar.radio("📍 Navigation", [
    "🏠 Price Prediction",
    "📊 Analysis",
    "📈 Model Evaluation",
    "👨‍💻 About"
])


# ---------------------------
# Common Helper Function
# ---------------------------
def safe_unique(col, default_list):
    if col in df.columns:
        vals = df[col].dropna().unique().tolist()
        if len(vals) > 0:
            return sorted(vals)
    return default_list

# ==========================================================
# 🏠 PRICE PREDICTION PAGE
# ==========================================================
if page == "🏠 Price Prediction":
    st.title("📱 Mobile Price Prediction")
    st.markdown("""
    Use the sidebar to enter phone specs and select a model, then click **Predict** to estimate the price.  
    Similar models from the dataset will be shown with their images and details.
    """)

    # ===============================
    # 🖼️ Display Default Image Logic
    # ===============================
    if "show_image" not in st.session_state:
        st.session_state["show_image"] = True

    if st.session_state["show_image"]:
        try:
            mobiles_img = Image.open("app/mobiles.png")
            st.image(mobiles_img, use_container_width=True, caption="Mobile Price Prediction Dashboard")
        except FileNotFoundError:
            st.warning("⚠️ 'mobiles.png' not found in your project folder.")

    # Sidebar: Model Selector
    st.sidebar.header("🧠 Select Prediction Model")
    model_choice = st.sidebar.selectbox(
        "Choose the model",
        ["XGBoost", "Stacking Ensemble", "Random Forest", "Linear Regression"]
    )

    model_paths = {
        "XGBoost": "app/xgboost_model.pkl",
        "Stacking Ensemble": "app/stacking_ensemble_model.pkl",
        "Random Forest": "app/random_forest_model.pkl",
        "Linear Regression": "app/linear_regression_model.pkl"
    }

    try:
        model = load_model(model_paths[model_choice])
        st.sidebar.success(f"✅ Loaded {model_choice}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Could not load model: {e}")
        model = None

    # ---------------------------
    # Sidebar Input Section
    # ---------------------------
    def safe_unique(col, default_list):
        if col in df.columns:
            vals = df[col].dropna().unique().tolist()
            if len(vals) > 0:
                return sorted(vals)
        return default_list

    brand_options = safe_unique("Brand", ["Infinix", "Samsung", "Vivo", "Xiaomi", "Oppo", "OnePlus"])
    gpu_options = safe_unique("gpu_category", ['Mali', 'IMG/PowerVR', 'Adreno', 'Xclipse', 'Immortalis'])
    os_options = sorted(
        pd.to_numeric(df["OS"].dropna().unique(), errors="coerce").astype(int).tolist()
    ) if "OS" in df.columns else [14]

    st.sidebar.markdown("### 📱 Device Info")
    brand = st.sidebar.selectbox("Brand", brand_options)

    if os_options:
        default_index = os_options.index(12) if 12 in os_options else 0
        os_version = st.sidebar.selectbox("OS Version", os_options, index=default_index)
    else:
        os_version = st.sidebar.selectbox("OS Version", [12])

    gpu = st.sidebar.selectbox("GPU Category", gpu_options)
    weight = st.sidebar.number_input("Weight (g)", min_value=80, max_value=400, value=int(df["weight_in_g"].median() if "weight_in_g" in df.columns else 195))

    st.sidebar.markdown("### ⚙️ Performance")
    ram_options = sorted(df["ram_gb"].dropna().unique().astype(int).tolist()) if "ram_gb" in df.columns else [2, 4, 6, 8, 12, 16]
    storage_options = sorted(df["storage_gb"].dropna().unique().astype(int).tolist()) if "storage_gb" in df.columns else [64, 128, 256, 512]
    ram = st.sidebar.selectbox("RAM (GB)", ram_options, index=ram_options.index(8) if 8 in ram_options else 0)
    storage = st.sidebar.selectbox("Storage (GB)", storage_options, index=storage_options.index(256) if 256 in storage_options else 0)

    st.sidebar.markdown("### 📸 Camera & Display")
    main_cam_options = sorted(df["main_camera_max_MP"].dropna().unique().astype(int).tolist()) if "main_camera_max_MP" in df.columns else [13, 50, 64, 108, 200]
    front_cam_options = sorted(df["front_camer_max_MP"].dropna().unique().astype(int).tolist()) if "front_camer_max_MP" in df.columns else [8, 16, 32, 50]
    refresh_options = sorted(df["refresh_rate_hz"].dropna().unique().astype(int).tolist()) if "refresh_rate_hz" in df.columns else [60, 90, 120, 144]

    main_cam = st.sidebar.selectbox("Main Camera (MP)", main_cam_options, index=main_cam_options.index(50) if 50 in main_cam_options else 0)
    front_cam = st.sidebar.selectbox("Front Camera (MP)", front_cam_options, index=front_cam_options.index(16) if 16 in front_cam_options else 0)
    refresh = st.sidebar.selectbox("Refresh Rate (Hz)", refresh_options, index=refresh_options.index(120) if 120 in refresh_options else 0)

    st.sidebar.markdown("### 🔋 Battery & Build")
    battery_options = sorted(df["battery_mAh"].dropna().unique().astype(int).tolist()) if "battery_mAh" in df.columns else [3000, 4000, 5000, 6000]
    battery = st.sidebar.selectbox("Battery (mAh)", battery_options, index=battery_options.index(5000) if 5000 in battery_options else 0)

    st.sidebar.markdown("### 📶 Connectivity & Display Type")
    band_5g = st.sidebar.selectbox("5G Support", ["Yes", "No"])
    nfc = st.sidebar.selectbox("NFC Support", ["Yes", "No"])
    amoled = st.sidebar.selectbox("AMOLED Display", ["Yes", "No"])
    fourk = st.sidebar.selectbox("4K or More Support", ["Yes", "No"])
    type_c = st.sidebar.selectbox("Type-C Support", ["Yes", "No"])

    # ---------------------------
    # Predict Button
    # ---------------------------
    if st.sidebar.button("🔮 Predict Price"):
        st.session_state["show_image"] = False  # Hide image when Predict is clicked

        if model is None:
            st.error("⚠️ No model loaded!")
        else:
            try:
                input_data = pd.DataFrame({
                    "Brand": [brand],
                    "OS": [os_version],
                    "weight_in_g": [weight],
                    "5G_Band": [band_5g],
                    "NFC": [nfc],
                    "gpu_category": [gpu],
                    "Amoled_display": [amoled],
                    "refresh_rate_hz": [refresh],
                    "storage_gb": [storage],
                    "ram_gb": [ram],
                    "main_camera_max_MP": [main_cam],
                    "4K_or_More_supportive": [fourk],
                    "front_camer_max_MP": [front_cam],
                    "Type_C_supportive": [type_c],
                    "battery_mAh": [battery]
                })

                predicted_price = model.predict(input_data)[0]
                st.success(f"💰 **Predicted Price:** Rs. {predicted_price:,.0f}")

                same_brand = df[df["Brand"] == brand].copy()
                for c in ["ram_gb", "storage_gb", "main_camera_max_MP", "battery_mAh"]:
                    same_brand[c] = pd.to_numeric(same_brand[c], errors="coerce").fillna(0)

                same_brand["dist"] = (
                    (same_brand["ram_gb"] - float(ram)).abs() +
                    (same_brand["storage_gb"] - float(storage)).abs() +
                    (same_brand["main_camera_max_MP"] - float(main_cam)).abs() +
                    (same_brand["battery_mAh"] - float(battery)).abs()
                )

                similar_models = same_brand.nsmallest(4, "dist")

                st.subheader("📸 Similar Models from Dataset")
                num_cols = 4
                for i in range(0, len(similar_models), num_cols):
                    cols = st.columns(num_cols)
                    for j, (_, row) in enumerate(similar_models.iloc[i:i+num_cols].iterrows()):
                        with cols[j]:
                            st.image(row.get("Image URL", ""), caption=row.get("Model_Title", "Unknown Model"), width=150)
                            st.markdown(f"""
                                <div style="border:1px solid #ddd; border-radius:10px; padding:8px; text-align:left; background-color:#36454F; color:white;">
                                    <b>Price:</b> Rs. {int(row['Price_PKR']):,}<br>
                                    <b>RAM:</b> {int(row['ram_gb'])} GB<br>
                                    <b>Storage:</b> {int(row['storage_gb'])} GB<br>
                                    <b>Main Cam:</b> {int(row['main_camera_max_MP'])} MP<br>
                                    <b>Front Cam:</b> {int(row['front_camer_max_MP']) if not pd.isna(row['front_camer_max_MP']) else '—'} MP<br>
                                    <b>Battery:</b> {int(row['battery_mAh'])} mAh<br>
                                    <b>Refresh Rate:</b> {int(row['refresh_rate_hz']) if not pd.isna(row['refresh_rate_hz']) else '—'} Hz
                                </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

    # ---------------------------
    # 🔁 Reset Button (Optional)
    # ---------------------------
    if st.sidebar.button("🔁 Reset Page"):
        st.session_state["show_image"] = True
        st.rerun()


# ==========================================================
# 📊 ANALYSIS PAGE
# ==========================================================
elif page == "📊 Analysis":
    st.title("📊 Mobile Data Analysis Dashboard")
    st.markdown("Explore patterns and trends in mobile pricing data.")

    # Average Price by Brand
    st.subheader("💰 Average Price by Brand")
    brand_avg = df.groupby("Brand")["Price_PKR"].mean().reset_index().sort_values("Price_PKR", ascending=False)
    fig1 = px.bar(brand_avg, x="Brand", y="Price_PKR", color="Brand", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)

  
# -----------------------
# Interactive Heatmap
# -----------------------
    st.subheader("🔥 Heatmap of Feature Correlations")

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu_r",
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        height=600,
        #title="Feature Correlation Heatmap",
        xaxis_nticks=len(corr.columns),
    )
    st.plotly_chart(fig, use_container_width=True)



    # -----------------------
    # Battery vs Price Analysis
    # -----------------------
    if "battery_mAh" in df.columns and "Price_PKR" in df.columns:
        st.subheader("🔋 Battery vs Price")
        battery_fig = px.scatter(
            df,
            x="battery_mAh",
            y="Price_PKR",
            color="Brand" if "Brand" in df.columns else None,
            size="ram_gb" if "ram_gb" in df.columns else None,
            hover_name="Brand",
            #title="How Battery Capacity Impacts Price",
            trendline="ols",
        )
        st.plotly_chart(battery_fig, use_container_width=True)


# -----------------------
# 3D Feature Interaction (Optional)
# -----------------------
    if all(col in df.columns for col in ["ram_gb", "storage_gb", "Price_PKR"]):
        st.subheader("🧭 3D Interaction: RAM, Storage, and Price")
        fig_3d = px.scatter_3d(
            df,
            x="ram_gb",
            y="storage_gb",
            z="Price_PKR",
            color="Brand" if "Brand" in df.columns else None,
            hover_data=df.columns,
            #title="3D View of Mobile Specs vs Price",
        )
        st.plotly_chart(fig_3d, use_container_width=True)



# ==========================================================
# 📈 MODEL EVALUATION PAGE
# ==========================================================
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation Report")
    st.markdown("""
    Evaluate and compare all trained models based on key regression metrics:
    - **R² (Coefficient of Determination)** → Higher is better  
    - **MAE (Mean Absolute Error)** → Lower is better  
    - **RMSE (Root Mean Squared Error)** → Lower is better  
    """)

    models = {
        "XGBoost": "xgboost_model.pkl",
        "Stacking Ensemble": "stacking_ensemble_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Linear Regression": "linear_regression_model.pkl"
    }

    if "Price_PKR" in df.columns:
        y = df["Price_PKR"]
        X = df[['Brand', 'OS', 'weight_in_g', '5G_Band', 'NFC', 'gpu_category',
                'Amoled_display', 'refresh_rate_hz', 'storage_gb', 'ram_gb',
                'main_camera_max_MP', '4K_or_More_supportive',
                'front_camer_max_MP', 'Type_C_supportive', 'battery_mAh']]

        # Split (same as during training)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = []
        for name, path in models.items():
            try:
                model = joblib.load(path)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    "Model": name,
                    "R²": round(r2, 4),
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2)
                })
            except Exception as e:
                st.error(f"❌ Error evaluating {name}: {e}")

        result_df = pd.DataFrame(results)

        st.subheader("📋 Model Evaluation Summary")
        st.dataframe(result_df, use_container_width=True)

        # ----------------------------
        # 📊 Bar Chart (MAE vs RMSE)
        # ----------------------------
        st.subheader("📊 Model Error Comparison (MAE vs RMSE)")

        melted = result_df.melt(id_vars="Model", value_vars=["MAE", "RMSE"],
                                var_name="Metric", value_name="Score")

        fig = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=True,
            title="Model Comparison Based on Error Metrics (MAE & RMSE)"
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Error Value",
            legend_title="Metric",
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # 🏆 Identify Best Model
        # ----------------------------
        st.subheader("🏆 Best Model Based on Overall Performance")

        # Rank by R² (descending), MAE (ascending), RMSE (ascending)
        ranked = result_df.copy()
        ranked["R2_rank"] = ranked["R²"].rank(ascending=False)
        ranked["MAE_rank"] = ranked["MAE"].rank(ascending=True)
        ranked["RMSE_rank"] = ranked["RMSE"].rank(ascending=True)
        ranked["Average_Rank"] = ranked[["R2_rank", "MAE_rank", "RMSE_rank"]].mean(axis=1)

        best_row = ranked.loc[ranked["Average_Rank"].idxmin()]
        best_model = best_row["Model"]

        st.success(f"🏅 **Best Overall Model:** {best_model}")
        st.markdown(f"""
        - **R²:** {best_row['R²']:.4f}  
        - **MAE:** {best_row['MAE']:.2f}  
        - **RMSE:** {best_row['RMSE']:.2f}
        """)

        # Optional: show ranking table
        st.markdown("### 📊 Detailed Ranking Table")
        st.dataframe(ranked.sort_values("Average_Rank"), use_container_width=True)

    else:
        st.error("⚠️ Column 'Price_PKR' not found in dataset.")
# ==========================================================
# 👨‍💻 ABOUT PAGE
# ==========================================================
elif page == "👨‍💻 About":
    st.title("👨‍💻 About the Developer")
    st.write("<p style='color:blue; font-size: 30px; font-weight: bold;'>Usama Munawar</p>", unsafe_allow_html=True)
    st.markdown("""
    **Data Scientist | MPhil Scholar | Machine Learning Enthusiast**  
    Passionate about transforming raw data into meaningful insights and intelligent systems.
    """)

    st.write("##### 🌍 Connect with me:")
    socials = {
        "GitHub": ("https://github.com/UsamaMunawarr", "https://img.icons8.com/fluent/48/000000/github.png"),
        "LinkedIn": ("https://www.linkedin.com/in/abu--usama", "https://img.icons8.com/color/48/000000/linkedin.png"),
        "YouTube": ("https://www.youtube.com/@CodeBaseStats", "https://img.icons8.com/?size=50&id=19318&format=png"),
        "Twitter": ("https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09", "https://img.icons8.com/color/48/000000/twitter.png"),
        "Facebook": ("https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO", "https://img.icons8.com/color/48/000000/facebook-new.png")
    }

    social_html = ""
    for name, (url, icon) in socials.items():
        social_html += f'<a href="{url}" target="_blank"><img src="{icon}" width="50" style="margin:5px;"></a>'
    st.markdown(social_html, unsafe_allow_html=True)

    st.write("<p style='color:green; font-size: 25px; font-weight: bold;'>Thank you for using this app, share it with your friends! 😊</p>", unsafe_allow_html=True)
