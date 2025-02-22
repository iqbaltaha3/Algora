import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, confusion_matrix, classification_report)

# --- Regression Algorithms ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# --- Classification Algorithms ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ------------------ Page Config & Custom CSS ------------------
st.set_page_config(page_title=" Algora ", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding: 2rem 3rem; }
    h1 { color: #2E4053; font-size: 3rem; font-weight: bold; text-align: center; }
    h2 { color: #2E4053; font-size: 2rem; font-weight: bold; border-bottom: 2px solid #2E4053; padding-bottom: 0.3rem; }
    h3 { color: #34495E; font-size: 1.75rem; font-weight: bold; }
    .css-1d391kg { font-size: 1.1rem; }
    .stButton > button { color: #fff; background-color: #2E86C1; border: none; border-radius: 0.5rem; padding: 0.5rem 1rem; font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
st.sidebar.title("Algora: Simplified ML training")
st.sidebar.info("Upload your dataset, configure your columns, and build models for Regression or Classification.")

task_type = st.sidebar.radio("Task Type", ["Regression", "Classification"])

# ------------------ App Title ------------------
st.title("Algora")
st.write(f"Build and compare ML models for **{task_type}** tasks with your own data.")

# ------------------ Data Upload ------------------
st.header("Data Upload")
st.write("Upload a CSV or Excel file containing your dataset.")
file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if file is None:
    st.info("Please upload your dataset to proceed.")
    st.stop()

try:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
except Exception as e:
    st.error(f"Error reading the file. ({e})")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------ Column Definition ------------------
st.header("Column Definition")
st.write("Select columns to remove (Primary Keys) and choose the target variable.")
try:
    primary_keys = st.multiselect("Primary Key Columns", df.columns, help="Optional: Unique identifier columns.")
    target_col = st.selectbox("Target Column", df.columns, help="Column to predict.")
    if primary_keys:
        df.drop(columns=primary_keys, inplace=True, errors='ignore')
    y = df[target_col].copy()
    df.drop(columns=[target_col], inplace=True, errors='ignore')
except Exception as e:
    st.error(f"Error defining columns. ({e})")
    st.stop()

# Preserve original features for prediction input
df_original = df.copy()

if task_type == "Classification":
    try:
        le = LabelEncoder()
        y = le.fit_transform(y)
        mapping = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}
        st.write("Target encoded. Mapping:", mapping)
    except Exception as e:
        st.error(f"Error encoding target variable. ({e})")
        st.stop()

# ------------------ Data Exploration ------------------
st.header("Data Exploration")
try:
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.markdown(f"**Categorical Columns:** {', '.join(cat_cols) if cat_cols else 'None'}")
    st.markdown(f"**Numerical Columns:** {', '.join(num_cols) if num_cols else 'None'}")
    st.subheader("Summary Statistics")
    st.write(df.describe())
    if len(num_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
except Exception as e:
    st.error(f"Error during exploration. ({e})")
    st.stop()

# ------------------ Data Preprocessing ------------------
st.header("Data Preprocessing")
st.write("Preprocessing includes missing value imputation, one-hot encoding for categoricals, and scaling for numericals.")
st.markdown(
    f"""
    - **Missing Values:**  
      - Numerical: {", ".join(num_cols) if num_cols else "None"} → filled with median  
      - Categorical: {", ".join(cat_cols) if cat_cols else "None"} → filled with "Unknown"
    - **Categorical Encoding:** One-hot encoding creates binary columns.
    - **Scaling:** Standard scaler is applied to numerical columns.
    """
)
try:
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna("Unknown", inplace=True)
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        df.drop(columns=cat_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, encoded_df], axis=1)
        joblib.dump(encoder, "encoder.pkl")
    scaler = StandardScaler()
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, "scaler.pkl")
    X = df.copy()
except Exception as e:
    st.error(f"Error during preprocessing. ({e})")
    st.stop()

# ------------------ Data Splitting ------------------
st.header("Data Splitting")
try:
    test_ratio = st.slider("Test Set Percentage", 10, 50, 30) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    st.write(f"Data split: {100 - test_ratio*100:.0f}% training, {test_ratio*100:.0f}% testing.")
except Exception as e:
    st.error(f"Error splitting data. ({e})")
    st.stop()

# ------------------ Model Training ------------------
st.header("Model Training")
st.write("Configure and train your models below.")
try:
    if task_type == "Regression":
        model_choices = st.multiselect("Regression Models", 
                                       ["Random Forest Regressor", "Linear Regression", "SVR", "Decision Tree Regressor"],
                                       default=["Random Forest Regressor", "Linear Regression"])
    else:
        model_choices = st.multiselect("Classification Models",
                                       ["Logistic Regression", "Random Forest Classifier", "SVC", "Decision Tree Classifier", "K-Nearest Neighbors"],
                                       default=["Random Forest Classifier", "Logistic Regression"])
    grid_search = st.checkbox("Use Grid Search for Hyperparameter Tuning", value=False)
    results = {}
    models_trained = {}
    residuals_store = {}

    for m in model_choices:
        st.subheader(m)
        if task_type == "Regression":
            if m == "Random Forest Regressor":
                if grid_search:
                    params = {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]}
                    rf = RandomForestRegressor(random_state=42)
                    grid = GridSearchCV(rf, params, cv=3, scoring="neg_mean_squared_error")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    n_estimators = st.slider("Number of Trees", 10, 200, 100, key="rf_reg")
                    max_depth = st.slider("Max Depth", 2, 20, 5, key="rf_reg_depth")
                    model_instance = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "Linear Regression":
                model_instance = LinearRegression()
                with st.spinner("Training model..."):
                    time.sleep(2)
                    model_instance.fit(X_train, y_train)
            elif m == "SVR":
                if grid_search:
                    params = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": [0.1, 1, 10]}
                    svr = SVR()
                    grid = GridSearchCV(svr, params, cv=3, scoring="neg_mean_squared_error")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svr_kernel")
                    C_val = st.slider("C Value", 0.1, 10.0, 1.0, key="svr_C")
                    model_instance = SVR(kernel=kernel, C=C_val)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "Decision Tree Regressor":
                if grid_search:
                    params = {"max_depth": [3, 5, 7, None]}
                    dt = DecisionTreeRegressor(random_state=42)
                    grid = GridSearchCV(dt, params, cv=3, scoring="neg_mean_squared_error")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    max_depth = st.slider("Max Depth", 2, 20, 5, key="dt_reg")
                    model_instance = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            mse_val = mean_squared_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)
            mae_val = mean_absolute_error(y_test, y_pred)
            results[m] = {"MSE": mse_val, "R2": r2_val, "MAE": mae_val}
            models_trained[m] = model_instance
            residuals_store[m] = y_test - y_pred
            st.write(f"Performance: MSE = {mse_val:.2f}, R² = {r2_val:.2f}, MAE = {mae_val:.2f}")
            with st.expander("View Prediction & Residual Plots"):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                axes[0].scatter(y_test, y_pred, alpha=0.7)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                axes[0].set_title("Actual vs Predicted")
                axes[0].set_xlabel("Actual")
                axes[0].set_ylabel("Predicted")
                sns.histplot(residuals_store[m], bins=20, kde=True, ax=axes[1])
                axes[1].set_title("Residual Distribution")
                st.pyplot(fig)
            if hasattr(model_instance, "feature_importances_"):
                imp = model_instance.feature_importances_
                feat_names = X_train.columns
                fi_df = pd.DataFrame({"Feature": feat_names, "Importance": imp}).sort_values(by="Importance", ascending=False)
                with st.expander("View Feature Importances"):
                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=fi_df.head(10), x="Importance", y="Feature", ax=ax_imp)
                    ax_imp.set_title("Top 10 Feature Importances")
                    st.pyplot(fig_imp)
            elif m == "Linear Regression" and hasattr(model_instance, "coef_"):
                coefs = model_instance.coef_
                feat_names = X_train.columns
                coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
                coef_df["Abs"] = coef_df["Coefficient"].abs()
                coef_df = coef_df.sort_values(by="Abs", ascending=False)
                with st.expander("View Model Coefficients"):
                    fig_coef, ax_coef = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=coef_df.head(10), x="Coefficient", y="Feature", ax=ax_coef)
                    ax_coef.set_title("Top 10 Coefficients")
                    st.pyplot(fig_coef)
        else:
            if m == "Logistic Regression":
                if grid_search:
                    params = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
                    lr = LogisticRegression(max_iter=1000)
                    grid = GridSearchCV(lr, params, cv=3, scoring="accuracy")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    C_val = st.slider("C Value", 0.1, 10.0, 1.0, key="lr_C")
                    model_instance = LogisticRegression(C=C_val, max_iter=1000)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "Random Forest Classifier":
                if grid_search:
                    params = {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]}
                    rf_clf = RandomForestClassifier(random_state=42)
                    grid = GridSearchCV(rf_clf, params, cv=3, scoring="accuracy")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    n_estimators = st.slider("Number of Trees", 10, 200, 100, key="rf_clf")
                    max_depth = st.slider("Max Depth", 2, 20, 5, key="rf_clf_depth")
                    model_instance = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "SVC":
                if grid_search:
                    params = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": [0.1, 1, 10]}
                    svc = SVC(probability=True)
                    grid = GridSearchCV(svc, params, cv=3, scoring="accuracy")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svc_kernel")
                    C_val = st.slider("C Value", 0.1, 10.0, 1.0, key="svc_C")
                    model_instance = SVC(kernel=kernel, C=C_val, probability=True)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "Decision Tree Classifier":
                if grid_search:
                    params = {"max_depth": [3, 5, 7, None]}
                    dt_clf = DecisionTreeClassifier(random_state=42)
                    grid = GridSearchCV(dt_clf, params, cv=3, scoring="accuracy")
                    with st.spinner("Grid searching..."):
                        grid.fit(X_train, y_train)
                    model_instance = grid.best_estimator_
                    st.write("Optimal parameters:", grid.best_params_)
                else:
                    max_depth = st.slider("Max Depth", 2, 20, 5, key="dt_clf")
                    model_instance = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                    with st.spinner("Training model..."):
                        time.sleep(2)
                        model_instance.fit(X_train, y_train)
            elif m == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn")
                model_instance = KNeighborsClassifier(n_neighbors=n_neighbors)
                with st.spinner("Training model..."):
                    time.sleep(2)
                    model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[m] = {"Accuracy": acc}
            models_trained[m] = model_instance
            st.write(f"Accuracy: {acc:.2f}")
            with st.expander("View Detailed Report"):
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title("Confusion Matrix")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
        st.markdown("---")
except Exception as e:
    st.error("Error during model training: " + str(e))
    st.stop()

# ------------------ Model Performance Comparison ------------------
st.header("Model Performance Comparison")
try:
    comp_df = pd.DataFrame(results).T
    st.subheader("Performance Metrics")
    st.dataframe(comp_df)
    st.subheader("Performance Visualization")
    fig_perf, ax_perf = plt.subplots(figsize=(8, 5))
    if task_type == "Regression":
        ax_perf.bar(comp_df.index, comp_df["MSE"], color='skyblue')
        ax_perf.set_title("MSE Comparison (Lower is Better)")
        ax_perf.set_ylabel("MSE")
    else:
        ax_perf.bar(comp_df.index, comp_df["Accuracy"], color='lightgreen')
        ax_perf.set_title("Accuracy Comparison (Higher is Better)")
        ax_perf.set_ylabel("Accuracy")
    ax_perf.set_xticklabels(comp_df.index, rotation=45)
    st.pyplot(fig_perf)
except Exception as e:
    st.error("Error comparing model performance: " + str(e))

# ------------------ Model Download ------------------
st.header("Download Your Model")
st.write("Select a model to download for future use.")
try:
    download_model = st.selectbox("Choose Model", list(models_trained.keys()))
    model_file = f"{download_model.replace(' ', '_').lower()}.pkl"
    joblib.dump(models_trained[download_model], model_file)
    with open(model_file, "rb") as f:
        st.download_button(label=f"Download {download_model}", data=f, file_name=model_file, mime="application/octet-stream")
except Exception as e:
    st.error("Error preparing model for download: " + str(e))

# ------------------ Advanced Prediction Section ------------------
st.header("Make Predictions")
st.write("Enter values for the original features (excluding primary keys and target) to obtain a prediction. For categorical features, choose from a dropdown; for numerical, enter a number.")

# Extract original features from df_original
original_num_cols = df_original.select_dtypes(include=['int64', 'float64']).columns.tolist()
original_cat_cols = df_original.select_dtypes(include=['object']).columns.tolist()

with st.form(key="predict_form"):
    st.subheader("Input Feature Values")
    input_data = {}
    for col in original_num_cols:
        input_data[col] = st.number_input(label=col, value=0.0, key=f"num_{col}")
    for col in original_cat_cols:
        options = list(df_original[col].dropna().unique())
        if not options:
            options = ["Unknown"]
        input_data[col] = st.selectbox(label=col, options=options, key=f"cat_{col}")
    submit_btn = st.form_submit_button(label="Predict")

if submit_btn:
    try:
        new_df = pd.DataFrame([input_data])
        # Apply same preprocessing as training
        if original_cat_cols:
            encoder = joblib.load("encoder.pkl")
            encoded_arr = encoder.transform(new_df[original_cat_cols])
            encoded_df = pd.DataFrame(encoded_arr, columns=encoder.get_feature_names_out(original_cat_cols))
            new_df.drop(columns=original_cat_cols, inplace=True)
            new_df.reset_index(drop=True, inplace=True)
            new_df = pd.concat([new_df, encoded_df], axis=1)
        if original_num_cols:
            scaler = joblib.load("scaler.pkl")
            new_df[original_num_cols] = scaler.transform(new_df[original_num_cols])
        # Use the first trained model for prediction (can be adjusted as needed)
        pred_model = models_trained[list(models_trained.keys())[0]]
        pred = pred_model.predict(new_df)
        if task_type == "Classification" and 'le' in locals():
            pred = le.inverse_transform(pred)
            result = f"Predicted Class: {pred[0]}"
        else:
            result = f"Predicted Value: {pred[0]:.2f}"
        st.markdown(f"<div style='text-align: center; font-size: 2.5rem; color: green;'>{result}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")


