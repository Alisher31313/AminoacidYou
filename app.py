import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Классификация аминокислот по Raman-спектрам", page_icon="🧬")
st.title("🧪 Классификация аминокислот по Raman-спектру")
@st.cache_resource

def load_model():
    return tf.keras.models.load_model("models/raman_cnn_model.h5")

model = load_model()

@st.cache_data
def load_encoder():
    labels = ['Gly', 'Leu', 'Phe', 'Trp']
    le = LabelEncoder()
    le.fit(labels)
    return le
le = load_encoder()
description = {
    "Gly": "Глицин — самая простая аминокислота",
    "Leu": "Лейцин — аминокислота с разветвлённой цепью",
    "Phe": "Фенилаланин — ароматическая аминокислота",
    "Trp": "Триптофан — аминокислота с индольным кольцом"
}

st.sidebar.header("📤 Загрузите Raman-спектр")
uploaded_file = st.sidebar.file_uploader("Выберите файл (.csv или .xlsx)", type=["csv", "xlsx"])
use_example = st.sidebar.button("📂 Использовать пример")


df = None

def read_uploaded_file(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка при чтении файла: {e}")
    return None

def read_example_file(path="example_spectrum.csv"):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.sidebar.error("❌ Файл примера не найден. Убедитесь, что example_spectrum.csv лежит в корне проекта.")
    except Exception as e:
        st.sidebar.error(f"❌ Ошибка загрузки примера: {e}")
    return None

df = None
if use_example:
    df = read_example_file()
    if df is not None:
        st.sidebar.success("✅ Примерный спектр успешно загружен.")
elif uploaded_file:
    df = read_uploaded_file(uploaded_file)
    if df is not None:
        st.sidebar.success("✅ Ваш файл успешно загружен.")

if df is not None:
    st.markdown("### 📈 Загруженные данные")
    st.dataframe(df.head())
    try:
        spectrum = df.values.flatten().astype("float32")
        spectrum = spectrum / np.max(spectrum)
        spectrum = np.expand_dims(spectrum, axis=(0, -1))  # (1, длина, 1)

        prediction = model.predict(spectrum)
        pred_class = np.argmax(prediction, axis=1)[0]
        pred_label = le.inverse_transform([pred_class])[0]
        confidence = np.max(prediction) * 100

        st.success(f"🧬 Обнаружено: **{pred_label}**")
        st.metric("📊 Уверенность", f"{confidence:.2f}%")
        st.info(f"**Описание:** {description[pred_label]}")

        st.markdown("### 🔬 Распределение вероятностей по классам")
        prob_df = pd.DataFrame(prediction, columns=le.classes_).T.rename(columns={0: "Probability"})
        st.bar_chart(prob_df)

    except Exception as e:
        st.error(f"❌ Ошибка обработки данных: {e}")

else:
    st.info("📁 Загрузите файл со спектром или используйте пример.")

st.markdown("""
<hr style="margin-top: 50px; border-top: 1px solid #444;" />
<div style='text-align: center; color: gray; font-size: small;'>
    🧬 DS Capstone Project 2025 | Классификация аминокислот по Raman-спектрам |
</div>
""", unsafe_allow_html=True)
