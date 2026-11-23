"""E-Commerce Customer Purchase Prediction Using Logistic Regression (ShopSmart).

Standalone version of the notebook. Run:  python ecommerce_purchase_prediction.py
Needs in the same folder: Ecommerce_Purchase_Prediction_DataSet.csv and iqr_capper.py
Outputs: figures/ and artifacts/ (model .pkl, cleaned csv, results.json)
"""
import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of opening windows


# E-Commerce Customer Purchase Prediction Using Logistic Regression
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve, precision_recall_curve,
                             average_precision_score, ConfusionMatrixDisplay)
import joblib
from iqr_capper import IQRCapper   # custom IQR outlier-capping transformer (shipped alongside)

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='Set2')
RANDOM_STATE = 42
DATA_PATH = 'Ecommerce_Purchase_Prediction_DataSet.csv'
FIG_DIR, ART_DIR = 'figures', 'artifacts'
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(ART_DIR, exist_ok=True)
RESULTS = {}

def savefig(name):
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/{name}.png', dpi=120, bbox_inches='tight')
    plt.show()


# 1. Data understanding & exploratory data analysis
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
display(df.head())
df.info()
display(df.describe().T)
RESULTS['n_rows'], RESULTS['n_cols'] = int(df.shape[0]), int(df.shape[1])


# 1.1 Target distribution
vc = df['Purchase'].value_counts().sort_index()
RESULTS['class_counts'] = {int(k): int(v) for k, v in vc.items()}
RESULTS['purchase_rate'] = float(df['Purchase'].mean())
fig, ax = plt.subplots(figsize=(5.5, 4))
sns.barplot(x=['0 - No Purchase', '1 - Purchase'], y=vc.values, ax=ax)
ax.bar_label(ax.containers[0], labels=[f'{v:,} ({v/len(df):.1%})' for v in vc.values])
ax.set_title('Target distribution: Purchase'); ax.set_ylabel('Customers')
savefig('01_target_distribution')


# 1.2 Missing values
miss = pd.DataFrame({'missing': df.isna().sum(), 'pct': (df.isna().mean()*100).round(2)})
miss = miss[miss.missing > 0].sort_values('missing', ascending=False)
display(miss)
RESULTS['missing'] = {k: int(v) for k, v in miss['missing'].items()}
RESULTS['total_missing_cells'] = int(df.isna().sum().sum())
RESULTS['rows_with_any_missing'] = int(df.isna().any(axis=1).sum())
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=miss['pct'], y=miss.index, ax=ax, color='#66c2a5')
ax.bar_label(ax.containers[0], fmt='%.1f%%'); ax.set_xlabel('% missing'); ax.set_title('Missing values by column')
savefig('02_missing_values')

# Is missingness related to the target? (checks whether 'missing' itself is informative)
rows = []
for c in miss.index:
    g = df.groupby(df[c].isna())['Purchase'].mean()
    rows.append({'column': c, 'purchase_rate_if_present': g.get(False), 'purchase_rate_if_missing': g.get(True)})
display(pd.DataFrame(rows).round(3))


# 1.3 Data-quality checks (duplicates, invalid and impossible values)
num_cols_all = ['Age','Session_Duration_Min','Pages_Viewed','Products_Viewed','Cart_Items','Cart_Value_INR',
                'Discount_Percent','Previous_Orders','Days_Since_Last_Visit','Email_Open_Rate','Customer_Rating','Returns_Last_Year']
cat_cols_all = ['Gender','Device_Type','Traffic_Source','Membership_Type','Payment_Method']
checks = {
 'duplicate rows': int(df.duplicated().sum()),
 'duplicate Customer_ID': int(df['Customer_ID'].duplicated().sum()),
 'Age < 18 or > 100': int(((df.Age < 18) | (df.Age > 100)).sum()),
 'Email_Open_Rate outside [0,1]': int(((df.Email_Open_Rate < 0) | (df.Email_Open_Rate > 1)).sum()),
 'Customer_Rating outside [1,5]': int(((df.Customer_Rating < 1) | (df.Customer_Rating > 5)).sum()),
 'Discount_Percent outside [0,100]': int(((df.Discount_Percent < 0) | (df.Discount_Percent > 100)).sum()),
 'negative values in any numeric column': int((df[num_cols_all] < 0).any(axis=1).sum()),
 'Pages_Viewed not a whole number': int((df.Pages_Viewed.dropna() % 1 != 0).sum()),
 'Age not a whole number': int((df.Age.dropna() % 1 != 0).sum()),
 'Cart_Items = 0 but Cart_Value > 0': int(((df.Cart_Items == 0) & (df.Cart_Value_INR > 0)).sum()),
}
RESULTS['quality_checks'] = checks
display(pd.Series(checks, name='count').to_frame())
print('Categorical levels:')
for c in cat_cols_all: print(f'  {c}:', df[c].dropna().unique().tolist())


# 1.4 Distributions of numeric features
fig, axes = plt.subplots(3, 4, figsize=(18, 10)); axes = axes.ravel()
for ax, c in zip(axes, num_cols_all):
    sns.histplot(df[c].dropna(), bins=40, kde=False, ax=ax, color='#66c2a5'); ax.set_title(c); ax.set_xlabel('')
savefig('03_numeric_distributions')


# 1.5 Outlier detection (IQR rule)
cont_cols = ['Age','Session_Duration_Min','Pages_Viewed','Cart_Value_INR','Discount_Percent','Days_Since_Last_Visit']
rows = []
for c in cont_cols:
    s = df[c].dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    rows.append({'feature': c, 'Q1': q1, 'Q3': q3, 'lower_fence': lo, 'upper_fence': hi,
                 'n_below': int((s < lo).sum()), 'n_above': int((s > hi).sum()),
                 'pct_outliers': round(100*((s < lo) | (s > hi)).mean(), 2), 'min': s.min(), 'max': s.max()})
outlier_tbl = pd.DataFrame(rows).round(2)
display(outlier_tbl)
RESULTS['outliers_raw'] = outlier_tbl.to_dict(orient='records')


# 1.6 Purchase rate by categorical features
fig, axes = plt.subplots(2, 3, figsize=(17, 9)); axes = axes.ravel()
base = df.Purchase.mean()
for ax, c in zip(axes, cat_cols_all):
    s = df[c].fillna('Missing')
    rate = df.groupby(s)['Purchase'].mean().sort_values()
    sns.barplot(x=rate.index, y=rate.values, ax=ax, color='#66c2a5')
    ax.bar_label(ax.containers[0], fmt='%.3f'); ax.axhline(base, ls='--', color='gray', lw=1)
    ax.set_ylim(0.5, 0.8); ax.set_title(f'Purchase rate by {c}'); ax.set_xlabel(''); ax.tick_params(axis='x', rotation=20)
axes[-1].axis('off')
savefig('04_purchase_rate_categorical')
cat_rates = {c: df.groupby(df[c].fillna('Missing'))['Purchase'].mean().round(4).to_dict() for c in cat_cols_all}
RESULTS['purchase_rate_by_category'] = cat_rates


# 1.7 Purchase rate vs numeric features
discrete = ['Cart_Items','Products_Viewed','Previous_Orders','Returns_Last_Year']
continuous = ['Age','Session_Duration_Min','Pages_Viewed','Cart_Value_INR','Discount_Percent','Days_Since_Last_Visit','Email_Open_Rate','Customer_Rating']
fig, axes = plt.subplots(3, 4, figsize=(19, 11)); axes = axes.ravel()
for ax, c in zip(axes, discrete + continuous):
    if c in discrete:
        g = df.groupby(c)['Purchase'].agg(['mean', 'count']); g = g[g['count'] >= 30]; labels = [str(i) for i in g.index]
    else:
        g = df.groupby(pd.qcut(df[c], 5, duplicates='drop'), observed=True)['Purchase'].agg(['mean', 'count'])
        labels = [f'{i.left:.1f}-{i.right:.1f}' for i in g.index]
    ax.plot(range(len(g)), g['mean'].values, marker='o', color='#fc8d62'); ax.axhline(base, ls='--', color='gray', lw=1)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(labels, rotation=25, fontsize=8); ax.set_title(c); ax.set_ylim(0.45, 0.9)
savefig('05_purchase_rate_numeric')


# 1.8 Correlation heatmap
corr = df[num_cols_all + ['Purchase']].corr()
fig, ax = plt.subplots(figsize=(11, 8.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, annot_kws={'size': 8}); ax.set_title('Correlation matrix')
savefig('06_correlation_heatmap')
tc = corr['Purchase'].drop('Purchase').sort_values()
RESULTS['corr_with_target'] = tc.round(4).to_dict()
display(tc.to_frame('corr_with_Purchase').round(3))


# Key EDA insights

# 2. Preprocessing
# Deterministic cleaning (not data-dependent): drop ID + duplicates; Pages_Viewed is a count -> round to whole pages
data = df.drop(columns='Customer_ID').drop_duplicates().copy()
data['Pages_Viewed'] = data['Pages_Viewed'].round()
X, y = data.drop(columns='Purchase'), data['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print('Train:', X_train.shape, ' Test:', X_test.shape)
print('Train purchase rate: %.3f | Test purchase rate: %.3f' % (y_train.mean(), y_test.mean()))
RESULTS['n_train'], RESULTS['n_test'] = int(len(X_train)), int(len(X_test))

cont_cols  = ['Age','Session_Duration_Min','Pages_Viewed','Cart_Value_INR','Discount_Percent','Days_Since_Last_Visit']  # impute -> cap outliers -> scale
other_num  = ['Products_Viewed','Cart_Items','Previous_Orders','Email_Open_Rate','Customer_Rating','Returns_Last_Year']  # impute -> scale
cat_cols   = ['Gender','Device_Type','Traffic_Source','Membership_Type','Payment_Method']                                # impute (mode) -> one-hot

preprocessor = ColumnTransformer([
    ('cont',  Pipeline([('impute', SimpleImputer(strategy='median')), ('cap', IQRCapper(1.5)), ('scale', StandardScaler())]), cont_cols),
    ('count', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), other_num),
    ('cat',   Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), cat_cols),
])
preprocessor.fit(X_train, y_train)
feature_names = [n.split('__', 1)[1] for n in preprocessor.get_feature_names_out()]
print(len(feature_names), 'model features after encoding:'); print(feature_names)


# 2.1 Outlier handling: before vs after capping (training data)
before = X_train[cont_cols]
after = pd.DataFrame(preprocessor.named_transformers_['cont'][:2].transform(X_train[cont_cols]), columns=cont_cols)
cap_tbl = pd.DataFrame({'lower_fence': preprocessor.named_transformers_['cont']['cap'].lower_,
                        'upper_fence': preprocessor.named_transformers_['cont']['cap'].upper_,
                        'n_values_capped': [int(((before[c] > hi) | (before[c] < lo)).sum()) for c, lo, hi in
                            zip(cont_cols, preprocessor.named_transformers_['cont']['cap'].lower_, preprocessor.named_transformers_['cont']['cap'].upper_)]},
                       index=cont_cols).round(2)
display(cap_tbl)
RESULTS['capping'] = cap_tbl.reset_index().rename(columns={'index': 'feature'}).to_dict(orient='records')
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for i, c in enumerate(cont_cols):
    sns.boxplot(y=before[c], ax=axes[0, i], color='#fc8d62'); axes[0, i].set_title(f'{c}\n(before)', fontsize=9); axes[0, i].set_ylabel('')
    sns.boxplot(y=after[c],  ax=axes[1, i], color='#66c2a5'); axes[1, i].set_title('(after capping)', fontsize=9); axes[1, i].set_ylabel('')
savefig('07_outliers_before_after')

clean = df.drop_duplicates().copy(); clean['Pages_Viewed'] = clean['Pages_Viewed'].round()
for c in cont_cols + other_num: clean[c] = clean[c].fillna(clean[c].median())
for c in cat_cols: clean[c] = clean[c].fillna(clean[c].mode()[0])
for c in cont_cols:
    q1, q3 = clean[c].quantile([.25, .75]); iqr = q3 - q1
    clean[c] = clean[c].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
clean.to_csv(f'{ART_DIR}/ecommerce_cleaned.csv', index=False)
print('Saved cleaned dataset:', clean.shape, '| missing cells left:', int(clean.isna().sum().sum()))


# 3. Model development
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --- Baseline: default Logistic Regression
baseline = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
baseline.fit(X_train, y_train)

# --- Tuned: grid search over regularisation strength, penalty and class weights (5-fold stratified CV, ROC-AUC)
pipe = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(solver='liblinear', max_iter=2000, random_state=RANDOM_STATE))])
param_grid = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100], 'clf__penalty': ['l1', 'l2'], 'clf__class_weight': [None, 'balanced']}
grid = GridSearchCV(pipe, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print('Best params :', grid.best_params_)
print('Best CV AUC : %.4f' % grid.best_score_)
cvres = pd.DataFrame(grid.cv_results_)[['param_clf__C','param_clf__penalty','param_clf__class_weight','mean_test_score','std_test_score','mean_train_score']]
display(cvres.sort_values('mean_test_score', ascending=False).head(8).round(4))
RESULTS['best_params'] = {k.replace('clf__', ''): (None if v is None else (float(v) if isinstance(v, (int, float)) else v)) for k, v in grid.best_params_.items()}
RESULTS['best_cv_auc'] = float(grid.best_score_)
RESULTS['cv_auc_std'] = float(cvres.sort_values('mean_test_score', ascending=False).iloc[0]['std_test_score'])


# 3.1 Multicollinearity check (VIF)
Xt = preprocessor.transform(X_train)
vif = []
for j, name in enumerate(feature_names):
    others = np.delete(Xt, j, axis=1)
    r2 = LinearRegression().fit(others, Xt[:, j]).score(others, Xt[:, j])
    vif.append((name, 1 / (1 - r2) if r2 < 1 else np.inf))
vif = pd.DataFrame(vif, columns=['feature', 'VIF']).sort_values('VIF', ascending=False).round(3)
display(vif.head(10))
RESULTS['vif_max'] = float(vif.VIF.max()); RESULTS['vif_top'] = vif.head(5).to_dict(orient='records')


# 4. Evaluation (hold-out test set)
def evaluate(model, Xa, ya, t=0.5):
    p = model.predict_proba(Xa)[:, 1]; pred = (p >= t).astype(int)
    return dict(accuracy=accuracy_score(ya, pred), precision=precision_score(ya, pred), recall=recall_score(ya, pred),
                f1=f1_score(ya, pred), roc_auc=roc_auc_score(ya, p), pr_auc=average_precision_score(ya, p))

rows = {'Baseline (train)': evaluate(baseline, X_train, y_train), 'Baseline (test)': evaluate(baseline, X_test, y_test),
        'Tuned (train)': evaluate(best, X_train, y_train), 'Tuned (test)': evaluate(best, X_test, y_test)}
comp = pd.DataFrame(rows).T.round(4); display(comp)
RESULTS['metrics'] = {k: {m: float(v) for m, v in r.items()} for k, r in rows.items()}
RESULTS['majority_baseline_accuracy'] = float(max(y_test.mean(), 1 - y_test.mean()))
print('Majority-class accuracy on test (naive benchmark): %.4f' % RESULTS['majority_baseline_accuracy'])
print('\nOverfitting check -> train AUC %.4f | CV AUC %.4f | test AUC %.4f' % (rows['Tuned (train)']['roc_auc'], grid.best_score_, rows['Tuned (test)']['roc_auc']))

proba_test = best.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)
print(classification_report(y_test, pred_test, target_names=['No Purchase', 'Purchase'], digits=3))
cm = confusion_matrix(y_test, pred_test)
RESULTS['confusion_matrix_0.5'] = cm.tolist()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=['No Purchase', 'Purchase']).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
ax.set_title('Confusion matrix (tuned model, threshold 0.5)'); ax.grid(False)
savefig('08_confusion_matrix')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, m in [('Baseline', baseline), ('Tuned', best)]:
    p = m.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, p):.3f})')
    pr, rc, _ = precision_recall_curve(y_test, p); axes[1].plot(rc, pr, label=f'{name} (AP = {average_precision_score(y_test, p):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random'); axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate'); axes[0].set_title('ROC curve'); axes[0].legend()
axes[1].axhline(y_test.mean(), color='k', ls='--', lw=1, label='No-skill'); axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('Precision-Recall curve'); axes[1].legend()
savefig('09_roc_pr_curves')


# 4.1 Threshold analysis
oof = cross_val_predict(best, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
grid_t = np.round(np.arange(0.05, 0.96, 0.01), 2)
f1s = [f1_score(y_train, oof >= t) for t in grid_t]
t_f1 = float(grid_t[int(np.argmax(f1s))])
fpr_o, tpr_o, thr_o = roc_curve(y_train, oof); t_youden = float(thr_o[int(np.argmax(tpr_o - fpr_o))])

def at(t):
    pred = (proba_test >= t).astype(int)
    return dict(threshold=round(t, 3), accuracy=accuracy_score(y_test, pred), precision=precision_score(y_test, pred, zero_division=0),
                recall=recall_score(y_test, pred), f1=f1_score(y_test, pred), flagged_pct=pred.mean())
thr_table = pd.DataFrame([at(0.5), at(t_f1), at(t_youden)], index=['Default 0.5', 'F1-optimal (CV)', 'Balanced / Youden (CV)']).round(3)
display(thr_table)
RESULTS['threshold_table'] = thr_table.reset_index().rename(columns={'index': 'strategy'}).to_dict(orient='records')
RESULTS['t_f1'], RESULTS['t_youden'] = t_f1, t_youden

sweep = pd.DataFrame([at(t) for t in np.arange(0.2, 0.86, 0.05)]).round(3); display(sweep)
RESULTS['threshold_sweep'] = sweep.to_dict(orient='records')
ts = np.arange(0.05, 0.96, 0.01)
curve = pd.DataFrame([at(t) for t in ts])
fig, ax = plt.subplots(figsize=(8.5, 5))
for m in ['precision', 'recall', 'f1', 'accuracy']: ax.plot(curve.threshold, curve[m], label=m)
ax.axvline(t_f1, color='gray', ls=':', label=f'F1-optimal = {t_f1:.2f}'); ax.axvline(0.5, color='k', ls='--', lw=1, label='Default 0.5')
ax.set_xlabel('Decision threshold'); ax.set_ylabel('Score (test set)'); ax.set_title('Metrics vs threshold'); ax.legend(ncol=2)
savefig('10_threshold_analysis')


# Reading the results

# 5. Interpretation: what drives purchases?
clf = best.named_steps['clf']
coef = pd.DataFrame({'feature': feature_names, 'coefficient': clf.coef_[0]})
coef['odds_ratio'] = np.exp(coef.coefficient); coef = coef.sort_values('coefficient', ascending=False).reset_index(drop=True)
display(coef.round(4))
RESULTS['coefficients'] = coef.round(4).to_dict(orient='records')
top = pd.concat([coef.head(10), coef.tail(10)]).drop_duplicates('feature')
top = top[top.coefficient != 0]
fig, ax = plt.subplots(figsize=(9, 7))
sns.barplot(data=top, y='feature', x='coefficient', ax=ax, palette=['#66c2a5' if v > 0 else '#fc8d62' for v in top.coefficient])
ax.axvline(0, color='k', lw=0.8); ax.set_title('Logistic Regression coefficients (tuned model)')
savefig('11_coefficients')


# Business reading of the drivers

# 6. Predicting purchase for new customers
os.makedirs(ART_DIR, exist_ok=True)
joblib.dump(best, f'{ART_DIR}/purchase_model_pipeline.pkl')
json.dump({'recommended_threshold_f1': t_f1, 'balanced_threshold': t_youden, 'default_threshold': 0.5}, open(f'{ART_DIR}/thresholds.json', 'w'), indent=2)
model = joblib.load(f'{ART_DIR}/purchase_model_pipeline.pkl')           # reload to prove the saved pipeline works end-to-end
assert np.allclose(model.predict_proba(X_test)[:, 1], proba_test)

new_customers = pd.DataFrame([
 # Engaged, loyal premium customer with a full cart
 dict(Age=32, Gender='Female', Device_Type='Mobile', Traffic_Source='Email', Membership_Type='Premium', Payment_Method='UPI',
      Session_Duration_Min=18.0, Pages_Viewed=12, Products_Viewed=7, Cart_Items=4, Cart_Value_INR=5200.0, Discount_Percent=15.0,
      Previous_Orders=6, Days_Since_Last_Visit=2.0, Email_Open_Rate=0.80, Customer_Rating=4.5, Returns_Last_Year=0),
 # Window-shopper: new member, short visit, empty cart
 dict(Age=24, Gender='Male', Device_Type='Desktop', Traffic_Source='Social Media', Membership_Type='New', Payment_Method='Wallet',
      Session_Duration_Min=3.0, Pages_Viewed=3, Products_Viewed=1, Cart_Items=0, Cart_Value_INR=400.0, Discount_Percent=5.0,
      Previous_Orders=0, Days_Since_Last_Visit=30.0, Email_Open_Rate=0.10, Customer_Rating=3.2, Returns_Last_Year=1),
 # Typical average customer
 dict(Age=34, Gender='Female', Device_Type='Mobile', Traffic_Source='Organic', Membership_Type='Regular', Payment_Method='Credit Card',
      Session_Duration_Min=10.8, Pages_Viewed=7, Products_Viewed=4, Cart_Items=1, Cart_Value_INR=1430.0, Discount_Percent=10.8,
      Previous_Orders=3, Days_Since_Last_Visit=8.5, Email_Open_Rate=0.50, Customer_Rating=3.8, Returns_Last_Year=0),
 # Lapsed customer with many returns
 dict(Age=45, Gender='Male', Device_Type='Tablet', Traffic_Source='Paid Ads', Membership_Type='New', Payment_Method='Cash on Delivery',
      Session_Duration_Min=5.0, Pages_Viewed=4, Products_Viewed=2, Cart_Items=0, Cart_Value_INR=900.0, Discount_Percent=8.0,
      Previous_Orders=1, Days_Since_Last_Visit=60.0, Email_Open_Rate=0.15, Customer_Rating=2.9, Returns_Last_Year=3),
 # Record with missing values and outlier-sized numbers (shows the pipeline copes with messy input)
 dict(Age=np.nan, Gender=None, Device_Type='Mobile', Traffic_Source=None, Membership_Type='Premium', Payment_Method='UPI',
      Session_Duration_Min=np.nan, Pages_Viewed=9, Products_Viewed=5, Cart_Items=2, Cart_Value_INR=60000.0, Discount_Percent=12.0,
      Previous_Orders=4, Days_Since_Last_Visit=400.0, Email_Open_Rate=0.6, Customer_Rating=np.nan, Returns_Last_Year=0),
])
new_customers.insert(0, 'Profile', ['Engaged premium', 'Window shopper', 'Average customer', 'Lapsed / high returns', 'Messy record (NaNs, outliers)'])
pnew = model.predict_proba(new_customers.drop(columns='Profile'))[:, 1]
out = new_customers[['Profile', 'Membership_Type', 'Cart_Items', 'Products_Viewed', 'Previous_Orders']].copy()
out['P(purchase)'] = pnew.round(3)
out['Pred @0.5'] = np.where(pnew >= 0.5, 'Purchase', 'No Purchase')
out[f'Pred @{t_f1:.2f} (F1-opt)'] = np.where(pnew >= t_f1, 'Purchase', 'No Purchase')
display(out)
RESULTS['new_customers'] = out.to_dict(orient='records')


# 7. Save results
def to_py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(RESULTS, open(f'{ART_DIR}/results.json', 'w'), indent=2, default=to_py)
print('Saved: model pipeline, thresholds, cleaned dataset, results.json, figures/')