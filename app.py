import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

@st.cache_resource
def load_model(path='weather_model.pkl'):
    return joblib.load(path)

model     = load_model()
label_map = {0: 'Tidak Hujan', 1: 'Hujan'}

st.title("🧭 Prediksi Cuaca dengan Random Forest")
st.markdown(
    "Masukkan 5 fitur meteorologi untuk memprediksi apakah akan **hujan** atau **tidak hujan**, ")


with st.form("input_form"):
    suhu        = st.number_input("Suhu (Celsius)",          value=25.0,  step=0.1)
    kelembaban  = st.number_input("Kelembaban (%)",          value=80.0,  step=0.1)
    wind_speed  = st.number_input("Kecepatan Angin (km/jam)", value=5.0,   step=0.1)
    tebal_awan  = st.number_input("Tebal Awan (meter)",       value=1000.0,step=1.0)
    tekanan_atm = st.number_input("Tekanan Atmosfer (hPa)",   value=1013.25,step=0.1)
    submitted   = st.form_submit_button("Predict")

if submitted:
    # 1) Input & prediksi
    df = pd.DataFrame(
        [[suhu, kelembaban, wind_speed, tebal_awan, tekanan_atm]],
        columns=[
            "Suhu (Celsius)",
            "Kelembaban (%)",
            "Kecepatan Angin (km/jam)",
            "Tebal Awan (meter)",
            "Tekanan Atmosfer (hPa)"
        ]
    )
    pred_num = model.predict(df)[0]
    st.success(f"**Hasil Prediksi:** {label_map[pred_num]}")

    # 2) Visualisasi pohon (dipotong sampai depth=2)
    st.markdown("---")
    st.subheader("🌳 Visualisasi Pohon")

    estimator = model.estimators_[0]
    fig, ax  = plt.subplots(figsize=(16, 6))  # Lebar 16, tinggi 6
    plot_tree(
        estimator,
        feature_names=df.columns,
        class_names=[label_map[0], label_map[1]],
        filled=True,
        rounded=True,
        impurity=False,
        max_depth=2,            # Batasi sampai dua level
        proportion=True,        # Ukuran node proporsional jumlah sampel
        fontsize=10,
        ax=ax
    )
    ax.set_title("Terusan Prediksi Cuaca", fontsize=14, pad=20)
    plt.tight_layout()
    st.pyplot(fig)
