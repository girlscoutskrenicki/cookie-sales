from flask import Flask, request, jsonify, url_for, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
import time
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from sklearn.linear_model import BayesianRidge
from scipy.stats import linregress
import os

warnings.simplefilter("ignore", category=RuntimeWarning)

app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://gsci-frontend.onrender.com"]}}, supports_credentials=True)




# -------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------
df = pd.read_csv('FinalCookieSales_2020_2024.csv')

# Rename columns to match expected format
df = df.rename(columns={
    'SU_Name': 'SU Name',
    'SU_Num': 'SU #'
})

df = df.drop(columns=['date'], errors='ignore')
df = df.dropna()
df = df[df['number_cases_sold'] > 0]

df['troop_id'] = df['troop_id'].astype(int)
df['period'] = df['period'].astype(int)
df['number_of_girls'] = df['number_of_girls'].astype(float)
df['number_cases_sold'] = df['number_cases_sold'].astype(float)
df['period_squared'] = df['period'] ** 2

# Normalize cookie types
normalized_to_canonical = {
    'adventurefuls': 'Adventurefuls',
    'dosidos': 'Do-Si-Dos',
    'samoas': 'Samoas',
    'smores': "S'mores",
    'tagalongs': 'Tagalongs',
    'thinmints': 'Thin Mints',
    'toffeetastic': 'Toffee-tastic',
    'trefoils': 'Trefoils',
    'lemonups': 'Lemon-Ups'
}

def normalize_cookie_type(raw_name):
    raw_lower = raw_name.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '', raw_lower)
    return normalized_to_canonical.get(slug, raw_name)

df['canonical_cookie_type'] = df['cookie_type'].apply(normalize_cookie_type)

# Add historical stats for interval clamping
stats = df.groupby(['troop_id', 'canonical_cookie_type'])['number_cases_sold'].agg(['min', 'max']).reset_index()
stats.columns = ['troop_id', 'canonical_cookie_type', 'historical_low', 'historical_high']
df = df.merge(stats, on=['troop_id', 'canonical_cookie_type'], how='left')

# -------------------------------
# TRAIN RIDGE TO GET RMSE FOR INTERVAL WIDTH
# -------------------------------
def run_ridge_interval_analysis():
    groups = df.groupby(['troop_id', 'canonical_cookie_type'])
    y_train_all, y_pred_all = [], []

    for (troop, cookie), group in tqdm(groups):
        group = group.sort_values('period')
        train = group[group['period'] <= 4]
        test = group[group['period'] == 5]
        if train.empty or test.empty:
            continue

        X_train = train[['period', 'number_of_girls']]
        y_train = train['number_cases_sold']
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        best_model = Ridge(alpha=1.0)
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_train_scaled)

        y_train_all.extend(y_train)
        y_pred_all.extend(y_pred)

    # Check if we have any data before calculating RMSE
    if len(y_train_all) == 0:
        print("Warning: No valid training data found for RMSE calculation. Using default RMSE of 10.0")
        rmse = 10.0
    else:
        rmse = np.sqrt(mean_squared_error(y_train_all, y_pred_all))
    
    app.config['OVERALL_RIDGE_RMSE'] = rmse
    print(f"Global RMSE for prediction interval: {rmse:.2f}")

run_ridge_interval_analysis()

# -------------------------------
# API ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template("home.html")

@app.route('/healthz')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/predict')
def predict_page():
    return render_template("index.html")

@app.route('/api/troop_ids')
def get_troop_ids():
    return jsonify(sorted(df['troop_id'].unique().tolist()))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get request parameters: troop_id and num_girls.
        req_data = request.get_json() or {}
        troop_id = str(req_data.get("troop_id", "")).strip()
        input_num_girls = float(req_data.get("num_girls", 0))
        if not troop_id or input_num_girls <= 0:
            return jsonify({"error": "Invalid troop_id or num_girls"}), 400

        # Re-load and clean the data.
        df_new = pd.read_csv('FinalCookieSales_2020_2024.csv')
        df_new.rename(columns={
            'date': 'year',
            'number_cases_sold': 'cases_sold',
            'number_of_girls': 'num_girls'
        }, inplace=True)
        df_new['year'] = df_new['year'].astype(int)
        df_new['troop_id'] = df_new['troop_id'].astype(str).str.strip()
        df_new['cookie_type'] = df_new['cookie_type'].str.strip().str.lower()

        # Determine the test year: the latest year available for this troop.
        troop_data = df_new[df_new['troop_id'] == troop_id]
        if troop_data.empty:
            return jsonify({"error": "No data for the specified troop"}), 404
        pred_year = int(troop_data['year'].max())

        # Filter test data to only include rows for the test year and troop.
        test = df_new[(df_new['year'] == pred_year) & (df_new['troop_id'] == troop_id)]
        if test.empty:
            return jsonify([])

        # Set parameters for prediction.
        lambda_grid = [0.1, 1, 5, 10, 50, 100]
        lambda_default = 10
        k_smooth = 5

        # Define a mapping for cookie images.
        cookie_images = {
            "adventurefuls": "ADVEN.png",
            "do-si-dos": "DOSI.png",
            "samoas": "SAM.png",
            "s'mores": "SMORE.png",
            "tagalongs": "TAG.png",
            "thin mints": "THIN.png",
            "toffee-tastic": "TFTAS.png",
            "trefoils": "TREF.png",
            "lemon-ups": "LMNUP.png"
        }

        # Helper to normalize cookie type strings.
        def normalize_cookie_type(raw):
            if not isinstance(raw, str):
                return raw
            c = raw.strip().lower()
            c = c.replace('\n', '').replace('’', "'")
            c = c.replace('–', '-').replace('--', '-')
            return c

        # Dictionaries to store clusters and candidate predictions.
        clusters_by_year = {}
        all_predictions = []  # Each record includes candidate_mse and cluster_std.
        preds_for_rmse = []   # For fallback overall error margin.

        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import KFold
        from kneed import KneeLocator
        from tqdm import tqdm
        import numpy as np

        # ----- CLUSTERING STEP -----
        # Use training data from 2020 up to (but not including) pred_year for this troop.
        train = df_new[(df_new['year'] >= 2020) & 
                       (df_new['year'] < pred_year) &
                       (df_new['troop_id'] == troop_id)]
        # Group by (year, SU #, cookie_type).
        grouped = list(train.groupby(['year', 'SU #', 'cookie_type']))
        for (yr, su, cookie), group in tqdm(grouped, desc=f"Clustering for {pred_year}", leave=False):
            valid = group[(group['cases_sold'] > 0) & (group['num_girls'] > 0)].copy()
            if valid.empty or len(valid) < 3:
                continue
            valid['pga'] = valid['cases_sold'] / valid['num_girls']
            X = valid[['pga']].values
            max_k = min(10, len(X))
            wcss = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
                wcss.append(kmeans.inertia_)
            try:
                knee = KneeLocator(range(1, max_k + 1), wcss, curve='convex', direction='decreasing')
                optimal_k = knee.knee if knee.knee is not None else 1
            except Exception:
                optimal_k = 1
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto").fit(X)
            valid['cluster'] = kmeans.predict(X)
            key = (pred_year, troop_id, cookie)
            if key not in clusters_by_year:
                clusters_by_year[key] = []
            clusters_by_year[key].append(valid[['cases_sold', 'num_girls']])

        # ----- PREDICTION STEP -----
        for (t, cookie), group_test in tqdm(test.groupby(['troop_id', 'cookie_type']),
                                              desc=f"Prediction for {pred_year}", leave=False):
            test_row = group_test.iloc[0]
            su_val = test_row.get("SU #", None)
            key_prefix = (pred_year, t, cookie)
            training_dfs = clusters_by_year.get(key_prefix, [])
            cluster_df = pd.concat(training_dfs, ignore_index=True) if training_dfs else pd.DataFrame()
            # Compute cluster-based std if available.
            cluster_std = cluster_df['cases_sold'].std() if not cluster_df.empty else None

            # Initialize candidate prediction variables.
            ridge_cluster_pred, mse_cluster = None, float('inf')
            ridge_troop_pred, mse_troop, lambda_cv = None, float('inf'), None
            lin_pred, mse_lin = None, float('inf')
            pga_last_pred, mse_pga_last = None, float('inf')
            pga_avg_pred, mse_pga_avg = None, float('inf')
            su_pred, mse_su = None, float('inf')

            # Candidate 1: Ridge with clustering.
            if not cluster_df.empty and len(cluster_df) >= 2:
                X = np.c_[np.ones(len(cluster_df)), cluster_df['num_girls'].values]
                y = cluster_df['cases_sold'].values.reshape(-1, 1)
                kf = KFold(n_splits=min(len(cluster_df), 5), shuffle=True, random_state=42)
                best_lambda = lambda_default
                best_mse = float('inf')
                for lam in lambda_grid:
                    mses = []
                    for train_idx, val_idx in kf.split(X):
                        X_tr, X_val = X[train_idx], X[val_idx]
                        y_tr, y_val = y[train_idx], y[val_idx]
                        I = np.eye(X.shape[1])
                        I[0, 0] = 0
                        beta = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                        y_val_pred = X_val @ beta
                        mses.append(mean_squared_error(y_val, y_val_pred))
                    avg_mse = np.mean(mses)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_lambda = lam
                alpha = len(cluster_df) / (len(cluster_df) + k_smooth)
                lambda_final = alpha * best_lambda + (1 - alpha) * lambda_default
                I = np.eye(X.shape[1])
                I[0, 0] = 0
                beta = np.linalg.inv(X.T @ X + lambda_final * I).dot(X.T @ y)
                ridge_cluster_pred = np.array([1, input_num_girls]) @ beta
                mse_cluster = mean_squared_error(y, X @ beta)

            # Candidate 2: Ridge on troop only.
            troop_hist = df_new[(df_new['troop_id'] == t) &
                                (df_new['cookie_type'] == cookie) &
                                (df_new['year'] < pred_year)]
            troop_hist = troop_hist[(troop_hist['cases_sold'] > 0) & (troop_hist['num_girls'] > 0)]
            n_train = len(troop_hist)
            if n_train > 1:
                X_troop = np.c_[np.ones(n_train), troop_hist['num_girls'].values]
                y_troop = troop_hist['cases_sold'].values.reshape(-1, 1)
                if n_train == 2:
                    best_mse = float('inf')
                    for lam in lambda_grid:
                        X_tr, X_val = X_troop[:1], X_troop[1:]
                        y_tr, y_val = y_troop[:1], y_troop[1:]
                        I = np.eye(X_troop.shape[1])
                        I[0, 0] = 0
                        beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                        y_pred_temp = X_val @ beta_temp
                        mse_val = mean_squared_error(y_val, y_pred_temp)
                        if mse_val < best_mse:
                            best_mse = mse_val
                            lambda_cv = lam
                elif n_train >= 3:
                    kf = KFold(n_splits=min(n_train, 3), shuffle=True, random_state=42)
                    best_mse = float('inf')
                    for lam in lambda_grid:
                        mses = []
                        for train_idx, val_idx in kf.split(X_troop):
                            X_tr, X_val = X_troop[train_idx], X_troop[val_idx]
                            y_tr, y_val = y_troop[train_idx], y_troop[val_idx]
                            I = np.eye(X_troop.shape[1])
                            I[0, 0] = 0
                            beta_temp = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                            y_pred_temp = X_val @ beta_temp
                            mses.append(mean_squared_error(y_val, y_pred_temp))
                        avg_mse = np.mean(mses)
                        if avg_mse < best_mse:
                            best_mse = avg_mse
                            lambda_cv = lam
                else:
                    lambda_cv = lambda_default

                if lambda_cv is not None:
                    alpha = n_train / (n_train + k_smooth)
                    lambda_final_troop = alpha * lambda_cv + (1 - alpha) * lambda_default
                    I = np.eye(X_troop.shape[1])
                    I[0, 0] = 0
                    beta = np.linalg.inv(X_troop.T @ X_troop + lambda_final_troop * I).dot(X_troop.T @ y_troop)
                    ridge_troop_pred = np.array([1, input_num_girls]) @ beta
                    mse_troop = mean_squared_error(y_troop, X_troop @ beta)
            else:
                ridge_troop_pred = None
                mse_troop = float('inf')

            # Candidate 3: Linear Regression.
            if n_train >= 2:
                model = LinearRegression().fit(troop_hist[['num_girls']], troop_hist['cases_sold'])
                lin_pred = model.predict([[input_num_girls]])[0]
                mse_lin = mean_squared_error(troop_hist['cases_sold'],
                                             model.predict(troop_hist[['num_girls']]))
            
            # Candidate 4: Last Year PGA Prediction.
            if not troop_hist.empty:
                last_year = troop_hist['year'].max()
                last_row = troop_hist[troop_hist['year'] == last_year].iloc[0]
                pga_last = last_row['cases_sold'] / last_row['num_girls']
                pga_last_pred = pga_last * input_num_girls
                mse_pga_last = mean_squared_error([last_row['cases_sold']],
                                                  [pga_last * last_row['num_girls']])
            
            # Candidate 5: Average PGA Prediction.
            if not troop_hist.empty:
                avg_pga = (troop_hist['cases_sold'] / troop_hist['num_girls']).mean()
                pga_avg_pred = avg_pga * input_num_girls
                mse_pga_avg = mean_squared_error(troop_hist['cases_sold'],
                                                 troop_hist['num_girls'] * avg_pga)
            
            # Candidate 6: SU-level Ridge without clustering.
            su_data = df_new[(df_new['SU #'] == test_row['SU #']) &
                             (df_new['cookie_type'] == cookie) &
                             (df_new['year'] < pred_year)]
            su_data = su_data[(su_data['cases_sold'] > 0) & (su_data['num_girls'] > 0)]
            if len(su_data) >= 3:
                X = np.c_[np.ones(len(su_data)), su_data['num_girls'].values]
                y = su_data['cases_sold'].values.reshape(-1, 1)
                kf = KFold(n_splits=min(len(su_data), 5), shuffle=True, random_state=42)
                best_lambda = lambda_default
                best_mse = float('inf')
                for lam in lambda_grid:
                    mses = []
                    for train_idx, val_idx in kf.split(X):
                        X_tr, X_val = X[train_idx], X[val_idx]
                        y_tr, y_val = y[train_idx], y[val_idx]
                        I = np.eye(X.shape[1])
                        I[0, 0] = 0
                        beta = np.linalg.inv(X_tr.T @ X_tr + lam * I).dot(X_tr.T @ y_tr)
                        y_val_pred = X_val @ beta
                        mses.append(mean_squared_error(y_val, y_val_pred))
                    avg_mse = np.mean(mses)
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_lambda = lam
                I = np.eye(X.shape[1])
                I[0, 0] = 0
                beta = np.linalg.inv(X.T @ X + best_lambda * I).dot(X.T @ y)
                su_pred = np.array([1, input_num_girls]) @ beta
                mse_su = mean_squared_error(y, X @ beta)
            
            # Choose the best candidate prediction.
            candidates = [
                ('cluster_ridge', ridge_cluster_pred, mse_cluster),
                ('troop_ridge', ridge_troop_pred, mse_troop),
                ('linreg', lin_pred, mse_lin),
                ('pga_last', pga_last_pred, mse_pga_last),
                ('pga_avg', pga_avg_pred, mse_pga_avg),
                ('su_ridge', su_pred, mse_su)
            ]
            valid_candidates = [(name, pred, err) for name, pred, err in candidates 
                                if pred is not None and not np.isnan(pred)]
            if valid_candidates:
                best_method, best_pred, best_mse = min(valid_candidates, key=lambda x: x[2])
                preds_for_rmse.append(best_pred)
                all_predictions.append({
                    "troop_id": troop_id,
                    "cookie_type": normalize_cookie_type(cookie),
                    "actual": test_row['cases_sold'],
                    "predicted": best_pred,
                    "method": best_method,
                    "candidate_mse": best_mse,
                    "cluster_std": cluster_std,
                    "su": test_row.get("SU #", None)
                })

        # Fallback: if no candidate predictions were generated, use the test data's PGA.
        if not all_predictions:
            for (t, cookie), group_test in test.groupby(['troop_id', 'cookie_type']):
                test_row = group_test.iloc[0]
                pga = test_row['cases_sold'] / test_row['num_girls']
                fallback_pred = pga * input_num_girls
                all_predictions.append({
                    "troop_id": troop_id,
                    "cookie_type": normalize_cookie_type(cookie),
                    "actual": test_row['cases_sold'],
                    "predicted": fallback_pred,
                    "method": "fallback_pga",
                    "candidate_mse": None,
                    "cluster_std": None,
                    "su": test_row.get("SU #", None)
                })

        # Compute the prediction interval for each final prediction using a chain:
        # 1. Use candidate-based standard error (sqrt(candidate_mse)).
        # 2. Else, use the cluster data's standard deviation.
        # 3. Else, use the SU-level standard deviation.
        # 4. Otherwise, fallback to overall RMSE.
        final_predictions = []
        for pred in all_predictions:
            cookie = pred["cookie_type"]
            predicted_val = float(pred["predicted"])
            candidate_mse = pred.get("candidate_mse", None)
            cluster_std = pred.get("cluster_std", None)
            su_val = pred.get("su", None)

            if candidate_mse is not None and candidate_mse > 0:
                candidate_std = np.sqrt(candidate_mse)
                interval_width = 1.96 * candidate_std
            elif cluster_std is not None and not np.isnan(cluster_std):
                interval_width = 1.96 * cluster_std
            else:
                if su_val is not None:
                    su_data_all = df_new[(df_new['SU #'] == su_val) & (df_new['cookie_type'] == cookie)]
                    su_std = su_data_all['cases_sold'].std()
                else:
                    su_std = None

                if su_std is not None and not np.isnan(su_std):
                    interval_width = 1.96 * su_std
                else:
                    if preds_for_rmse:
                        overall_rmse = np.sqrt(mean_squared_error(preds_for_rmse, preds_for_rmse))
                        interval_width = overall_rmse * 2 if overall_rmse > 0 else 10
                    else:
                        interval_width = 10

            final_predictions.append({
                "cookie_type": cookie,
                "predicted_cases": round(predicted_val, 2),
                "interval_lower": round(max(1, predicted_val - interval_width), 2),
                "interval_upper": round(predicted_val + interval_width, 2),
                "image_url": url_for('static', filename=cookie_images.get(cookie, "default.png"), _external=True)
            })

        return jsonify(final_predictions)

    except Exception as e:
        print("Error in /api/predict:", e)
        return jsonify({"error": str(e)}), 500






# In app.py, modify /api/history endpoint:
@app.route('/api/history/<int:troop_id>')
def get_history(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify({"error": "No data"}), 404

    sales = troop_df.groupby('period')['number_cases_sold'].sum().reset_index()
    girls = troop_df.groupby('period')['number_of_girls'].mean().reset_index()

    su = None
    suName = None
    if 'SU #' in troop_df.columns and 'SU Name' in troop_df.columns:
        su = int(troop_df['SU #'].iloc[0])
        suName = troop_df['SU Name'].iloc[0]

    return jsonify({
        "totalSalesByPeriod": [{"period": int(r['period']), "totalSales": r['number_cases_sold']} for _, r in sales.iterrows()],
        "girlsByPeriod": [{"period": int(r['period']), "numberOfGirls": r['number_of_girls']} for _, r in girls.iterrows()],
        "su": su,
        "suName": suName
    })


@app.route('/api/cookie_breakdown/<int:troop_id>')
def get_breakdown(troop_id):
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify([])

    grouped = troop_df.groupby(['period', 'canonical_cookie_type'])['number_cases_sold'].sum().reset_index()
    pivoted = grouped.pivot(index='period', columns='canonical_cookie_type', values='number_cases_sold').fillna(0)
    pivoted.reset_index(inplace=True)

    return jsonify(pivoted.to_dict(orient='records'))


@app.route('/api/su_search')
def su_search():
    query = request.args.get('q', '').strip()
    if not query.isdigit():
        return jsonify([])

    matches = df[df['SU #'].astype(str).str.startswith(query)]
    results = (
        matches[['SU #', 'SU Name']]
        .drop_duplicates()
        .sort_values('SU #')
        .to_dict(orient='records')
    )
    return jsonify(results)


@app.route('/api/su_history/<int:su_num>')
def su_history(su_num):
    df_su = df[df['SU #'] == su_num]
    if df_su.empty:
        return jsonify({"error": "No data"}), 404

    # Average number of girls per year across all troops
    girls_by_year = df_su.groupby('period')['number_of_girls'].mean().reset_index()

    # Step 1: Get total cases sold per troop per year (sum over cookie types)
    troop_sales = df_su.groupby(['period', 'troop_id'])['number_cases_sold'].sum().reset_index()

    # Step 2: Get average total sales per troop per year
    sales_by_year = troop_sales.groupby('period')['number_cases_sold'].mean().reset_index()
    sales_by_year.rename(columns={'number_cases_sold': 'avgSales'}, inplace=True)

    # Scatter plot data (per cookie type)
    scatter = df_su[['number_of_girls', 'number_cases_sold', 'canonical_cookie_type']].dropna().to_dict(orient='records')

    return jsonify({
        "girlsByYear": [
            {"period": int(r['period']), "avgGirls": r['number_of_girls']}
            for _, r in girls_by_year.iterrows()
        ],
        "salesByYear": [
            {"period": int(r['period']), "avgSales": r['avgSales']}
            for _, r in sales_by_year.iterrows()
        ],
        "scatterData": scatter
    })



@app.route('/api/su_scatter_regression/<int:su_num>')
def su_scatter_regression(su_num):
    from scipy.stats import linregress
    
    # Filter data for that SU and remove rows with missing values
    filtered = df[df['SU #'] == su_num].dropna(subset=['number_of_girls', 'number_cases_sold'])
    if filtered.empty or filtered['number_of_girls'].nunique() < 2:
        # Not enough data to compute regression
        return jsonify({"line": [], "lower": [], "upper": []})
    
    # Optional: remove outliers from 'number_cases_sold'
    q1 = filtered['number_cases_sold'].quantile(0.25)
    q3 = filtered['number_cases_sold'].quantile(0.75)
    iqr = q3 - q1
    filtered = filtered[
        (filtered['number_cases_sold'] >= q1 - 1.5 * iqr) &
        (filtered['number_cases_sold'] <= q3 + 1.5 * iqr)
    ]
    
    # If everything got filtered out, return empty
    if filtered.empty or filtered['number_of_girls'].nunique() < 2:
        return jsonify({"line": [], "lower": [], "upper": []})
    
    # Run linear regression
    x = filtered['number_of_girls']
    y = filtered['number_cases_sold']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Build line arrays
    x_vals = sorted(set(x))
    line = []
    lower = []
    upper = []
    for xi in x_vals:
        pred = slope * xi + intercept
        margin = 2 * std_err  # you can tweak the multiplier
        line.append({"x": xi, "y": pred})
        lower.append({"x": xi, "y": pred - margin})
        upper.append({"x": xi, "y": pred + margin})
    
    return jsonify({"line": line, "lower": lower, "upper": upper})

@app.route('/api/regression/<int:troop_id>')
def regression(troop_id):
    # Filter data for the given troop ID
    troop_df = df[df['troop_id'] == troop_id]
    if troop_df.empty:
        return jsonify({"error": "No data found for troop"}), 404

    # Extract x and y values
    x = troop_df['number_of_girls']
    y = troop_df['number_cases_sold']

    # Perform a simple linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Create regression line points over the range of x
    x_min, x_max = x.min(), x.max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = slope * x_vals + intercept

    # Compute a simple confidence band: ±2 * std_err
    margin = 2 * std_err
    lower_band = y_vals - margin
    upper_band = y_vals + margin

    # Combine band data into a single array
    band_data = []
    for xv, lb, ub in zip(x_vals, lower_band, upper_band):
        band_data.append({
            "number_of_girls": float(xv),
            "lower": float(lb),
            "upper": float(ub)
        })

    # Prepare scatter data points from the raw data
    scatter_data = [
        {"number_of_girls": float(ng), "number_cases_sold": float(cs)}
        for ng, cs in zip(x, y)
    ]

    # Prepare regression line data points
    line_data = [
        {"number_of_girls": float(x_val), "number_cases_sold": float(y_val)}
        for x_val, y_val in zip(x_vals, y_vals)
    ]

    return jsonify({
        "scatter": scatter_data,
        "regression_line": line_data,
        "band": band_data
    })
@app.route('/api/regression/<int:su_num>')
def regression_su(su_num):
    # Filter data for the given SU number
    su_df = df[df['SU #'] == su_num]
    if su_df.empty:
        return jsonify({"error": "No data found for SU"}), 404

    # Extract x and y values for regression
    x = su_df['number_of_girls']
    y = su_df['number_cases_sold']

    # Ensure there is enough variation in x for regression
    if x.nunique() < 2:
        return jsonify({"error": "Not enough data to perform regression"}), 400

    # Run linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Generate regression line data over the range of x values
    x_min, x_max = x.min(), x.max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = slope * x_vals + intercept

    # Compute a confidence band: ± 2×std_err
    margin = 2 * std_err
    lower_band = y_vals - margin
    upper_band = y_vals + margin

    # Build the confidence band data points
    band_data = []
    for xv, lb, ub in zip(x_vals, lower_band, upper_band):
        band_data.append({
            "number_of_girls": float(xv),
            "lower": float(lb),
            "upper": float(ub)
        })

    # Prepare the scatter (raw) data points
    scatter_data = [
        {"number_of_girls": float(ng), "number_cases_sold": float(cs)}
        for ng, cs in zip(x, y)
    ]

    # Prepare the regression line data points
    regression_line = [
        {"number_of_girls": float(x_val), "number_cases_sold": float(y_val)}
        for x_val, y_val in zip(x_vals, y_vals)
    ]

    return jsonify({
        "scatter": scatter_data,
        "regression_line": regression_line,
        "band": band_data
    })

@app.route('/api/su_predict', methods=['POST'])
def su_predict():
    try:
        data = request.get_json()
        print("📦 Received data:", data)
        su_num = int(data.get('su_number'))
        num_girls = float(data.get('num_girls'))
        pred_year = 5

        df_su = df[
            (df['SU #'] == su_num) &
            (df['number_of_girls'] > 0) &
            (df['number_cases_sold'] > 0) &
            (df['period'] < pred_year)  # Prevent data leakage
        ].copy()

        if df_su.empty:
            return jsonify([])

        predictions = []
        cookie_images = {
            "Adventurefuls": "ADVEN.png",
            "Do-Si-Dos": "DOSI.png",
            "Samoas": "SAM.png",
            "S'mores": "SMORE.png",
            "Tagalongs": "TAG.png",
            "Thin Mints": "THIN.png",
            "Toffee-tastic": "TFTAS.png",
            "Trefoils": "TREF.png",
            "Lemon-Ups": "LMNUP.png"
        }

        for cookie_type in df_su['canonical_cookie_type'].unique():
            cookie_df = df_su[df_su['canonical_cookie_type'] == cookie_type]
            if len(cookie_df) < 3:
                continue

            X = np.c_[np.ones(len(cookie_df)), cookie_df['number_of_girls'].values]
            y = cookie_df['number_cases_sold'].values

            model = BayesianRidge(fit_intercept=False)
            model.fit(X, y)

            X_pred = np.array([[1, num_girls]])
            y_pred, std = model.predict(X_pred, return_std=True)

            lower = max(1, y_pred[0] - 1.96 * std[0])
            upper = y_pred[0] + 1.96 * std[0]
            pred_val = float(y_pred[0])

            image_url = url_for('static', filename=cookie_images.get(cookie_type, "default.png"), _external=True)

            predictions.append({
                "cookie_type": cookie_type,
                "predicted_cases": round(pred_val, 2),
                "interval_lower": round(lower, 2),
                "interval_upper": round(upper, 2),
                "image_url": image_url
            })

        return jsonify(predictions)
    except Exception as e:
        print("❌ ERROR in /api/su_predict:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
