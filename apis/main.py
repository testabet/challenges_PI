import fastapi
from typing import Union
from fastapi import FastAPI,  HTTPException
from pydantic import BaseModel
from schemas import DocumentInput, UploadResponse, EmbeddingsResponse,SearchResponse, ItemsResponse, AskResponse
from typing import List
import hashlib
from rag import load_key, generate_chunks, get_embeddings, generacion_rta
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
import logging

DOC_DB: List[dict] =[]
found_docs= []

# Crea la clase personalizada de EmbeddingFunction para ChromaDB
class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Llama a la función de Cohere para obtener las embeddings
        return get_embeddings(input)  # input es una lista de textos


# Configuracion de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#Iniciacion de FastAPI y Chromadb
app = FastAPI()
client = chromadb.Client()
collection = client.get_or_create_collection(name="documentos",
                                      embedding_function=MyEmbeddingFunction())



@app.post("/upload/",response_model=UploadResponse,status_code=201)
def upload_file(doc:DocumentInput):
    """Carga un nuevo documento y le designa ID alfanumerico. """
    
    #Verifica que el titulo y el contenido no este vacio
    if not doc.title.strip() or not doc.content.strip():
        logger.warning(f"Intento de guardar un documento vacio o sin titulo: {doc}")
        raise HTTPException(status_code=422, detail="No fue posible agrergar el documento")
    
    # genera ID alfanumérico
    new_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()

    # verifica que el documento no se encuentre cargado
    for existing_doc in DOC_DB:
        if existing_doc["id"] == new_id:
             logger.info(f"Documento encontrado: {new_id}")
             return {"message": "El documento fue encontrado en la base de datos",
                    "document_id": new_id}
    
    # guarda el documento en la base de datos
    doc_dic= doc.model_dump() 
    doc_dic["id"]=new_id 
    DOC_DB.append(doc_dic)
    
    
    logger.info(f"Documento cargado: {new_id} ")
    return {"message": "El documento fue cargado correctamente",
                    "document_id": new_id}

  
@app.post("/generate-embeddings/{document_id}",response_model=EmbeddingsResponse,status_code=201)
def generate_embeddings(document_id:str):
    """Genera los embeddings del documento correspondiente al ID brindado"""
    #verifica que el ID no sea un campo vacio
    if not document_id.strip():
        logger.warning(f"Intento de generar embeddings para ID inexistente: {document_id}")
        raise HTTPException(status_code=422, detail="ID invalido") 
    
    for doc in DOC_DB:
        
        if doc["id"]==document_id:
            logger.info(f"Iniciando generación de embeddings para: {document_id}")
            
            # genera los chunks
            documento_dividido=generate_chunks(doc["content"])
            
            # Crea metadatos con los titulos y ID del documento
            metadatas = []
            for _ in  documento_dividido:
                metadatas.append({
                        "title": doc["title"],
                        "document_id": document_id 
                            })

            #genera los ids y embeddings de cada chunk y lo guarda en la base de datos
            ids,embeddings= get_embeddings(documento_dividido)
          
            #guarda embeddings en DB
            try:
                collection.add(
                    documents=documento_dividido,
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                    )
                logger.info(f"Se generaron {len(documento_dividido)} embeddigns para el documento {document_id}")
            
            except Exception as e:
                logger.error(f"Error guardando en ChromaDB: {e}")
                raise HTTPException(status_code=500, detail="Error interno al guardar vectores.")
        
            
            return {"message": f"Embedding generado correctamente para el documento",
                    "document_id":f"{document_id}"}
    
    raise HTTPException(status_code=402, detail="ID no encontrado para generar el embedding")


@app.post("/search/{query}",response_model=SearchResponse,status_code=201) 
def search_docs(query:str):
    if not query.strip():
        logger.warning(f"Consulta invalida: {query}")
        raise HTTPException(status_code=422, detail="Consulta invalida") 

    logger.info(f"Búsqueda recibida: {query}")
    id_query,query_emb=get_embeddings([query])
    

    # Consultar sobre la base
    try:
        results = collection.query(
                query_embeddings=query_emb,
                n_results=3, # cantidad de resultados seleccionada tras ir realizando pruebas con diferentes valores 
                include=["documents", "metadatas", "distances"]
                )
        logger.info(f"Consulta generada")
    
    except Exception as e:
        logger.error(f"Error en ChromaDB al realizar la consulta: {e}")
        raise HTTPException(status_code=500, detail="Error en servicio de embeddings al realizar la consulta.")
    
    #recorro la rta para obtener lo que me importa
    found_docs = []


    if results["ids"] and results["ids"][0]:
        for id_, doc, meta, dist in zip(
            results['ids'][0], 
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ):
            score= 1-dist 
            is_grounded = True if score >= 0.1 else False
            
            parent_id = meta.get("document_id", "ID_desconocido") 
            title = meta.get("title", "Sin título")

            item = ItemsResponse(
                        document_id=parent_id,
                        title=title,
                        content_snippet=doc, # El chunk de texto encontrado
                        id_embedding=id_,
                        similarity_score=score, # Convertir distancia a score 
                        grounded= is_grounded)
                    
            found_docs.append(item)
   
    return {"results":found_docs}


@app.post("/ask/{question}",response_model=AskResponse,status_code=201)
def aks_llm(question:str):
    logger.info(f"Pregunta recibida {question}")
    found_docs=search_docs(question)
    
    #Revisa las similitudes, si la maxima tiene menos de 0.1 se fuerza a una respuesta
    scores= [d.similarity_score for d in found_docs["results"]]
    if max(scores)<0.1:
        respuesta_final= AskResponse(question= question,
                                    answer="No cuento con informacion suficiente para responder a esta consulta",
                                    context_used=[],
                                    similarity_score= [],
                                    grounded= [False])
        logger.info(f"Respuesta forzada por falta de contexto")
        return respuesta_final

    try:
        respuesta= generacion_rta(found_docs, question)
        logger.info(f"Respuesta generada")
    except Exception as e:
        return {"error":"EL servicio externo no pudo procesar la solicitud en este momento"}
        logger.error(f"Error en Cohere al generar la respuesta: {e}")
        #raise HTTPException(status_code=500, detail="Error en servicio de Cohere.")


    context_used_aux=[] 
    similarity_score_aux=[]
    grounded_aux=[]
    
    for d in found_docs["results"]:
        context_used_aux.append(d.content_snippet)
        similarity_score_aux.append(d.similarity_score)
        grounded_aux.append(d.grounded)
        
    
    respuesta_final= AskResponse(question= question,
                                    answer=respuesta,
                                    context_used= context_used_aux,
                                    similarity_score= similarity_score_aux,
                                    grounded= grounded_aux)
    
    return respuesta_final

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True) # esto se usa en desarrollo
                            # 0.0.0.0 para levantar un serivdos y si no funciona cambiarlo a 127.0.0.1
    # la otra forma le damos el comando directo cuando lo usamos para produccion,
    #  para eso se debe actiavr el entorno y se lo damos desde el comando