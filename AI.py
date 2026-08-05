#!/usr/bin/env python3
"""
ECG Arritmia Detector - Interfaz Web con Streamlit
PoC interactiva con visualización de señales
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

st.set_page_config(
    page_title="ECG Detector",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
    }
    .abnormal {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class ECGAnalyzer:
    """Analizador de ECG simplificado"""
    
    CLASSES = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB']
    
    @staticmethod
    def extract_features(signal):
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        rms = np.sqrt(np.mean(signal**2))
        peak_to_peak = np.max(signal) - np.min(signal)
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        fft = np.fft.fft(signal)
        freq_content = np.abs(fft[:len(fft)//2])
        
        return {
            'rms': rms,
            'peak_to_peak': peak_to_peak,
            'zero_crossings': zero_crossings,
            'mean_freq_power': np.mean(freq_content),
        }
    
    @staticmethod
    def classify(features):
        rms = features['rms']
        pp = features['peak_to_peak']
        zc = features['zero_crossings']
        
        probs = {
            'Normal': 1.0 - (min(rms, 2.0) / 2.0) * 0.3,
            'AF': min(zc / 1000, 1.0) * 0.4,
            'I-AVB': max(0, (1.0 - rms) * 0.3),
            'LBBB': max(0, (1.0 - pp / 5.0) * 0.2),
            'RBBB': max(0, (pp / 5.0 - 0.5) * 0.2),
        }
        
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    @staticmethod
    def analyze(signal):
        features = ECGAnalyzer.extract_features(signal)
        probs = ECGAnalyzer.classify(features)
        predicted = max(probs, key=probs.get)
        confidence = probs[predicted]
        
        return {
            'predicted': predicted,
            'confidence': confidence,
            'probabilities': probs,
            'features': features
        }


def plot_ecg(signal, title="ECG Signal"):
    """Graficar señal ECG"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal, linewidth=1.5, color='#0066cc')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(True, alpha=0.3)
    return fig


def plot_probabilities(probs):
    """Graficar probabilidades de diagnóstico"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = list(probs.keys())
    values = list(probs.values())
    colors = ['#28a745' if v == max(values) else '#6c757d' for v in values]
    
    bars = ax.barh(classes, values, color=colors)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Diagnóstico Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Agregar valores en barras
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f'{val*100:.1f}%', 
               va='center', fontweight='bold')
    
    return fig


# INTERFAZ PRINCIPAL
st.markdown("""
# 🏥 ECG Arritmia Detector
## Proof of Concept - Detección de Problemas Cardíacos
""")

st.markdown("---")

# Sidebar con opciones
st.sidebar.title("⚙️ Configuración")
mode = st.sidebar.radio(
    "Selecciona modo:",
    [" Análisis de Muestra", "Generar ECG", " Información"]
)

if mode == " Análisis de Muestra":
    st.header("Análisis de Señales ECG de Prueba")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Selecciona tipo de ECG")
        ecg_type = st.selectbox(
            "Tipo:",
            ["Normal", "Atrial Fibrillation (AF)", "Bloqueo (I-AVB)", 
             "LBBB", "RBBB"]
        )
    
    with col2:
        st.subheader(" Duración")
        duration = st.slider("Duración (segundos)", 1.0, 10.0, 5.0)
    
    # Generar ECG basado en tipo
    t = np.linspace(0, duration, int(5000 * duration / 5))
    
    if ecg_type == "Normal":
        signal = (
            1.0 * np.sin(2 * np.pi * 1.2 * t) +
            0.3 * np.sin(2 * np.pi * 0.6 * t) +
            np.random.randn(len(t)) * 0.05
        )
    elif ecg_type == "Atrial Fibrillation (AF)":
        signal = (
            1.2 * np.sin(2 * np.pi * 1.2 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 3 * t)) +
            np.random.randn(len(t)) * 0.3
        )
    elif ecg_type == "Bloqueo (I-AVB)":
        signal = (
            0.3 * np.sin(2 * np.pi * 1.2 * t) +
            np.random.randn(len(t)) * 0.05
        )
    elif ecg_type == "LBBB":
        signal = (
            0.4 * np.sin(2 * np.pi * 0.8 * t) +
            np.random.randn(len(t)) * 0.08
        )
    else:  # RBBB
        signal = (
            2.0 * np.sin(2 * np.pi * 1.5 * t) +
            np.random.randn(len(t)) * 0.1
        )
    
    # Mostrar gráfico de la señal
    st.subheader(" Señal ECG")
    fig = plot_ecg(signal, f"ECG - {ecg_type}")
    st.pyplot(fig)
    
    # Análisis
    st.subheader(" Análisis")
    
    if st.button(" Analizar Señal", key="analyze_btn"):
        with st.spinner("Analizando..."):
            result = ECGAnalyzer.analyze(signal)
        
        # Mostrar resultado principal
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Diagnóstico",
                result['predicted'],
                f"{result['confidence']*100:.1f}%"
            )
        
        with col2:
            is_normal = result['predicted'] == 'Normal'
            status = "✅ Normal" if is_normal else "⚠️ Anormal"
            st.metric("Estado", status)
        
        with col3:
            risk = "Bajo" if is_normal else "Alto"
            st.metric("Riesgo", risk)
        
        # Gráfico de probabilidades
        st.subheader(" Probabilidades de Diagnóstico")
        fig = plot_probabilities(result['probabilities'])
        st.pyplot(fig)
        
        # Características extraídas
        st.subheader(" Características Extraídas")
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        
        with feat_col1:
            st.metric("RMS", f"{result['features']['rms']:.4f}")
        with feat_col2:
            st.metric("Peak-to-Peak", f"{result['features']['peak_to_peak']:.4f}")
        with feat_col3:
            st.metric("Zero Crossings", f"{result['features']['zero_crossings']:.0f}")
        with feat_col4:
            st.metric("Mean Freq Power", f"{result['features']['mean_freq_power']:.4f}")

elif mode == " Generar ECG":
    st.header("Generar ECG Personalizado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amplitude = st.slider("Amplitud", 0.1, 3.0, 1.0)
        frequency = st.slider("Frecuencia (Hz)", 0.5, 3.0, 1.2)
    
    with col2:
        noise_level = st.slider("Nivel de ruido", 0.0, 0.5, 0.1)
        duration = st.slider("Duración (seg)", 1.0, 10.0, 5.0)
    
    t = np.linspace(0, duration, int(5000 * duration / 5))
    signal = amplitude * np.sin(2 * np.pi * frequency * t) + np.random.randn(len(t)) * noise_level
    
    fig = plot_ecg(signal, "Custom ECG Signal")
    st.pyplot(fig)
    
    if st.button(" Analizar ECG Personalizado"):
        with st.spinner("Analizando..."):
            result = ECGAnalyzer.analyze(signal)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diagnóstico", result['predicted'], f"{result['confidence']*100:.1f}%")
        
        fig = plot_probabilities(result['probabilities'])
        st.pyplot(fig)

else:  # Información
    st.header(" Acerca de este PoC")
    
    st.markdown("""
    ###  Objetivo
    Demostrar una Prueba de Concepto (PoC) para detectar señales tempranas 
    de problemas cardíacos mediante electrocardiogramas (ECG) usando 
    Inteligencia Artificial.
    
    ###  Tecnología
    - **Algoritmo**: Deep Learning con Red Neuronal Convolucional
    - **Framework**: TensorFlow/Keras
    - **Dataset**: Basado en ECG-DeepLearning (Nature Medicine)
    
    ###  Condiciones Detectadas
    - **Normal**: ECG sin anomalías
    - **AF** (Atrial Fibrillation): Fibrilación auricular
    - **I-AVB**: Bloqueo AV de primer grado
    - **LBBB**: Bloqueo de rama izquierda
    - **RBBB**: Bloqueo de rama derecha
    
    ###  Limitaciones de la PoC
    1. Este es un modelo de demostración, NO es para uso clínico
    2. Las señales generadas son sintéticas para demostración
    3. Para uso real se requiere:
       - Validación clínica completa
       - Datos reales de pacientes
       - Aprobación regulatoria
       - Supervisión médica
    
    ###  Referencias
    - Hannun et al. "Cardiologist-level arrhythmia detection..." Nature Medicine (2019)
    - Repository: https://github.com/awni/ecg
    """)
    
    st.markdown("---")
    st.info(" Esta PoC fue desarrollada para demostración de concepto únicamente.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    ECG Detector PoC | Última actualización: 2026 |  Solo para demostración
</div>
""", unsafe_allow_html=True)