import streamlit as st
import requests
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Asistente Clínico RAG para Hipertensión Arterial",
    page_icon="🩺",
    layout="centered"
)

# buscamos la variable de entorno API_URL, si no existe usamos la direccion de localhost
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🩺 Asistente de guías clínicas de Hipertensión Arterial (2024) y  Diabetes Mellitus Tipo 2 (2019)")
st.warning("⚠️**IMPORTANTE:**\nEsta es una herramienta de apoyo para las decisiones clinicas basadas en guías nacionales oficailes. " \
"NO reemplaza el juicio clínico profesional. Verifique siempre la fuente original.")
st.markdown("---")

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.info("Esta herramienta utiliza RAG (Retrieval-Augmented Generation) para responder preguntas sobre la **Guía de Práctica " \
    "Clínica Nacionalsobre Prevención, Diagnóstico y Tratamiento de la Hipertensión Arterial (HTA) | Actualizada 2024** y " \
    "**Guía de Práctica Clínica Nacional sobre Prevención, Diagnóstico y Tratamiento de la Diabetes Mellitus Tipo 2 (DM2) (2019)**.")
    
    # Botón para limpiar historial
    if st.button("Borrar chat"):
        st.session_state.messages = []
        st.rerun()


# --- GESTIÓN DEL ESTADO (HISTORIAL DE CHAT) ---
# Inicializamos la memoria de la sesión si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MOSTRAR HISTORIAL EN PANTALLA ---
# Cada vez que la app se actualiza, redibuja los mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si el mensaje tiene fuentes guardadas, las mostramos en un desplegable
        if "sources" in message and message["sources"]:
            with st.expander("📚 Ver Evidencia Consultada"):
                for fuente in message["sources"]:
                    st.markdown(f"- {fuente}")

# --- CAPTURA DE LA PREGUNTA DEL USUARIO ---
if prompt := st.chat_input("Escriba su consulta clínica aquí..."):
    
    #  Guardar y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara y envia el historial
    historial_a_enviar = [
    {"role": m["role"], "content": m["content"]}
    for m in st.session_state.messages]

    payload = {
    "question": prompt,
    "historial": historial_a_enviar}

    #  Llamar a la API (Backend)
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            try:
                # Enviamos la petición POST a FastAPI
                response = requests.post(API_URL, json=payload)
                
                if response.status_code != 201:
                    st.error(f"Error Código: {response.status_code}")
                    st.write(response.text) # Esto imprimirá el mensaje exacto de error de FastAPI
                
                if response.status_code == 201:
                    data = response.json()
                    respuesta_texto = data["answer"]
                    fuentes = data["evidence"] 
                    
                    # Mostrar respuesta
                    st.markdown(respuesta_texto)
                    
                    # Mostrar fuentes (Evidencia)
                    if fuentes:
                        with st.expander("📚 Ver Evidencia Consultada"):
                            for f in fuentes:
                                st.markdown(f"- {f}")
                    
                    # 3. Guardar respuesta en el historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": respuesta_texto,
                        "sources": fuentes
                    })
                    
                else:
                    error_msg = f"Error en el servidor: {response.status_code}"
                    st.error(error_msg)
            
            except requests.exceptions.ConnectionError:
                st.error("❌ No se pudo conectar con la API")