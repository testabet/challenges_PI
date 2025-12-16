import os
import json
import chromadb
import cohere
from dotenv import load_dotenv
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Definicion de funciones a usar 

def load_key():
    load_dotenv()  # Load .env file
    api_key = os.getenv("COHERE_API_KEY")
    print(api_key)  # Verify the key is loaded


def generate_chunks(docs:str):
    length_function = len

    splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"," ",""],
            chunk_size=300, 
            chunk_overlap=50,
            length_function=length_function
            )

    splits=splitter.split_text(docs)

    return splits

def get_embeddings(documentos_divididos:List[str]):
    """Genera embeddings"""
    load_key()
    co = cohere.ClientV2()
    response = co.embed(
        texts=documentos_divididos,
        model="embed-multilingual-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )
    embeddings=response.embeddings.float_

    ids= [f"id_{i}" for i in range (len(embeddings))]
 

    return ids,embeddings

def generacion_rta(context:str, question:str):
    load_key()
    co = cohere.ClientV2()
    
    system_promt=f"""Sos un asistente de lectura con capacidades de interpretar historias. Responde de manera amigable y con tono entusiasta como si le hablaras a un niño. 
    Responde en maximo 3 oraciones. Agrega emojis al final de la respuesta. Responde exclusivamente en español rioplatense. 
    Contesta la pregunta unicamente basandote en el siguiente contexto: {context}. Si no hay información suficiente devolver "No cuento con información suficiente para responder a 
    esta consulta". No brindar información sensible que no se encuentre explicitamente en los documentos cargados. No incluir estereotipos, insultos o juicios subjetivos."""             
    

    response = co.chat(
        model= "command-a-03-2025",
        messages=[{"role": "system", "content": {system_promt}},
                {"role": "user", "content": {question}}],
        temperature=0.1,
        seed=1212
        )
    
    return response.message.content[0].text
