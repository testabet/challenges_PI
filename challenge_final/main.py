import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import logging
from schemas import ResponseLLM, UserAsk
from collections import deque

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

  
def load_key():
    load_dotenv()  # Load .env file
    api_key = os.getenv("COHERE_API_KEY")
    if api_key:
        logger.info(f"La API KEY fue cargada correctamente")
    else:
        logger.warning(f"La API KEY no pudo cargarse o es incorrecta")

# --------------------- CONFIGURACIONES --------------------------------------------------------

# FastAPI
app = FastAPI()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Variable de entorno, base de datos e historial
load_key()
CHROMA_PATH = "./chroma_db"
chat_history=deque(maxlen=6)   


# Carga la base de datos usando el mismo modelo de embedding  
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings)

#---------------------------- Definicion de LLM, prompts y orquestador -----------------------------------
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        seed=1212
    )

system_prompt = (
        "Eres Clin, un asistente experto de apoyo a la toma de decisiones clínicas . "
        "Estas diseñado para asistir a médicos de atención primaria." 
        "Utiliza un tono formal y profesional. "
        "Utiliza lenguaje técnico, preciso y profesional basado estrictamente en las guías nacionales de Hipertensión Arterial y  Diabetes Mellitus Tipo 2.."
        "No simplifiques la terminología médica."
        "Usa los siguientes fragmentos de contexto recuperado para responder la pregunta. "
        "Si no sabes la respuesta basándote en el contexto, di que no lo sabes. "
        "NO inventes información. "
    
        "\n\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. Responde SIEMPRE en español rioplatense.\n"
        "2. NO uses emojis bajo ninguna circunstancia.\n"
        "3. Sé conciso, profesional y directo.\n"
        
        "\n\n"
        "{context}")

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{chat_history}"),
                            ]
    )

# junta los documentos recuperados y los manda al llm como contexto
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# orquestador
router_system_prompt = (
    "Eres un sistema de clasificación de intenciones para un asistente clinico. "
    "Tu ÚNICO trabajo es analizar la pregunta del usuario y clasificarla en una de estas 4 categorías:\n"
    "1. 'HTA': Si la pregunta está relacionada con Hipertensión Arterial, HTA, Recomendaciones de la guia clinica de hipertension arterial.\n"
    "2. 'DIABETES': Si la pregunta está relacionada con la Diabetes Mellitus Tipo 2, Diabetes, DMT2, recomendaciones de la guia clinica de Diabetes Mellitus Tipo 2.\n"
    "3. 'SALUDO': Si el usuario solo está saludando (ej: 'Hola', 'Buenos días', 'Gracias').\n"
    "4. 'OTRO': Si la pregunta es sobre cualquier otro tema no médico (ej: deportes, cocina, programación, clima).\n"
    "\n"
    "Responde EXCLUSIVAMENTE con una de esas cuatro palabras: 'HTA','DIABETES', 'SALUDO' u 'OTRO'. No digas nada más."
)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{input}"),
])

#definicion de la cadena
router_chain = router_prompt | llm | StrOutputParser()

# ---------------------- ENDPOINT --------------------------------------------------------------------
@app.post("/ask",response_model=ResponseLLM,status_code=201)
def consultar(question:UserAsk):
    
    if not question.question.strip():
        logger.warning(f"Consulta invalida: {question.question}")
        raise HTTPException(status_code=422, detail="Consulta invalida") 
    
    # deteccion de intención y entrada a cada tipo de respuesta
    try:
        logger.info(f"Analizando intención de: {question.question}")
        intencion = router_chain.invoke({"input": question.question}).strip().upper()
        logger.info(f"Intención detectada: {intencion}")

        # Preparamos el filtro vacío por defecto (busca en todo)
        filtro_tema = {} 

        if "HTA" in intencion:
            # Configurar filtro SOLO para HTA
            filtro_tema = {"tema": "HTA"} 
            
        elif "DIABETES" in intencion:
            # Configurar filtro SOLO para Diabetes
            filtro_tema = {"tema": "DIABETES"}
        print(filtro_tema)
        print(intencion)

    except Exception as e:
        logger.error(f"Error del seridor al detectar la intención de la consulta")
        raise HTTPException(status_code=500, detail=str(e))

    if "SALUDO" in intencion:
            logger.info(f"Se detecto un saludo")
            return {
                "answer": "¡Hola! Soy tu asistente especializado en Hipertensión Arterial y Diabetes Mellitus Tipo 2. ¿En qué puedo ayudarte hoy? Recuerda especificar la patología",
            }
    elif "OTRO" in intencion:
            logger.info(f"Se detecto una pregunta fuera del contexto")
            return {
                "answer": "Lo siento, como asistente clínico solo estoy autorizado para responder consultas sobre información "
                " que se encuentren en las guías referidas a la Hipertensión Arterial y la Diabetes Mellitus Tipo 2"
                ""
            }
    elif "HTA" in intencion or "DIABETES" in intencion:
        logger.info(f"Se detecto una pregunta relacionada a la HTA o a DMT2")
        # Consulta a la base de datos
        try:     
            results = vectorstore.similarity_search_with_score(question.question, k=8,filter=filtro_tema)
            logger.info(f"Búsqueda realizada: {question.question}")

        except Exception as e:
            logger.error(f"Error al realizar la consulta a la base de datos: {e}")
            raise HTTPException(status_code=500, detail="Error en servicio al realizar la consulta a la base de datos.")
    
        # Guarda los documentos cuya similitud sea menor a 0.7 ya que puntajes mas bajos representan mayor similitud
        # Si no se pasa este criterio, se devuelve una respuesta forzada
        UMBRAL_CONFIANZA = 0.8 
        
        docs_aprobados = []
        evidencia= []
        no_aprobado=[]
        
        for doc, score in results:
            if score <= UMBRAL_CONFIANZA:
                
                docs_aprobados.append(doc)
                
                # Guardamos esto para mostrarlo en la API o consola
                evidencia.append({
                    "text": doc.page_content[:100] + "...",
                    "score": round(1 - score, 2), # Para que valores mas altos representen mayor similitud
                    "title": doc.metadata.get("title","N/A"),
                    "pag": doc.metadata.get("page", "N/A")                
                })
            else:
                no_aprobado.append({
                    "text": doc.page_content[:100] + "...",
                    "score": round(1 - score, 2),  #Para que valores mas altos representen mayor similitud
                    "title": doc.metadata.get("title","N/A"),
                    "pag": doc.metadata.get("page", "N/A")            
                })
        logger.info(f"Hay {len(docs_aprobados)} fragmentos recuperados")

        if not docs_aprobados:
            logger.info(f"Respuesta forzada por falta de contexto")
            return {
                "answer": "Lo siento, no encontré información suficientemente relevante en los manuales para responder esa consulta con seguridad.",
                "evidence": no_aprobado 
                }
        
        #Recupero el historial en caso de que no este vacio y llamo al LLM
        chat_history=[]
        try: 
            if question.chat_history:
                memory_buffer= deque(question.chat_history, maxlen=6)
            
                for msg in memory_buffer:
                    if msg.role == "user":
                        chat_history.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        chat_history.append(AIMessage(content=msg.content))
            
            logger.info(f"Historial recuperado de las ultimas 3 preguntas/respuestas")

            respuesta_llm = question_answer_chain.invoke({"input": question.question,"context":docs_aprobados,"chat_history":chat_history})
            logger.info(f"Respuesta generada con LLM")
            
            return {"answer": respuesta_llm,
                    "evidence": evidencia}

        except Exception as e:
            logger.error(f"Error al generar la respuesta {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    else:
        logger.info(f"No se pudo clasificar la consulta")
        return {"answer": "No pude entender tu consulta. Por favor intenta preguntar sobre hipertensión arterial o la diabetes mellitus tipo 2."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)






