import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix

@st.cache_resource
def load_model():
    model = pickle.load(open('credit_model.pkl', 'rb'))
    return model


@st.cache_data
def load_data():
    data = pickle.load(open('test_data.pkl', 'rb'))
    return data['X_test'], data['y_test'], data['feature_names']


model = load_model()
X_test, y_test, feature_names = load_data()

st.title("👤 Проверка клиента")
st.markdown("Введите ID клиента, чтобы узнать решение модели и причины.")

client_id = st.number_input(
    "ID клиента (0 — первый, {} — последний):".format(len(X_test) - 1),
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)

client_data = X_test.iloc[client_id:client_id+1]
true_label = y_test.iloc[client_id]

proba = model.predict_proba(client_data)[0, 1]
threshold = 0.1585
decision = "🔴 ОТКАЗАТЬ" if proba >= threshold else "🟢 ОДОБРИТЬ"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Вероятность дефолта", f"{proba:.1%}")
with col2:
    st.metric("Порог", f"{threshold:.1%}")
with col3:
    st.metric("Решение", decision)

st.caption(f"Реальный класс: {'дефолт' if true_label == 1 else 'хороший'}")

st.subheader("🔍 Почему модель приняла такое решение?")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(client_data)

fig, ax = plt.subplots(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=client_data.iloc[0],
        feature_names=feature_names
    ),
    max_display=10,
    show=False
)
ax.set_title(f"SHAP Waterfall: ID = {client_id}, P(def) = {proba:.3f}", fontsize=14)
st.pyplot(fig)
plt.close()

st.subheader("📊 Топ-5 признаков, повлиявших на решение")

shap_impact = pd.DataFrame({
    'Признак': feature_names,
    'Значение': client_data.iloc[0].values,
    'SHAP (влияние)': shap_values[0]
})
shap_impact['|SHAP|'] = shap_impact['SHAP (влияние)'].abs()
top5 = shap_impact.sort_values('|SHAP|', ascending=False).head(5)

for i, row in top5.iterrows():
    direction = "🔴 повышает риск" if row['SHAP (влияние)'] > 0 else "🟢 снижает риск"
    st.write(f"**{row['Признак']}** = {row['Значение']:.4f} → {direction} (SHAP = {row['SHAP (влияние)']:+.4f})")