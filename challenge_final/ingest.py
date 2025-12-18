import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# ------------- CONFIGURACIÓN --------------------
# Carga las variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print("API KEY cargada con exito")


# Nombre de la carpeta donde se guardará la base de datos (NO borrar esta carpeta después)
CHROMA_PATH = "./chroma_db"


def main():
    "Cargo el manual"
    PDF_PATH = "guia-nacional-practica-clinica-diabetes-mellitius-tipo2_2019.pdf" 
    TEMA_ETIQUETA = "DIABETES"
    
    print(f"1. Cargando el archivo: {PDF_PATH}...")
    # Usamos PyMuPDF porque lee mejor las tablas y el formato que PyPDFLoader
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"   -> Se cargaron {len(docs)} páginas.")

    # Recorro cada pagina y le agrego la etiqueta manual
    for doc in docs:
        # Agregamos o sobrescribimos metadata
        doc.metadata["tema"] = TEMA_ETIQUETA
        doc.metadata["titulo"] = "Guía Nacional Practica Clinica de Diabetes Mellitus Tipo 2 (2019)" 
        
    
    print("2. Dividiendo texto (Chunking)...")
    # El chunk_size de 1300 es un buen balance para manuales técnicos.
    # El overlap de 200 ayuda a no cortar frases importantes entre fragmentos.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1300,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # Intenta separar por párrafos primero
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   -> Se crearon {len(chunks)} fragmentos de texto.")

    #print("3. Creando Base de Datos Vectorial (esto puede tardar un poco)...")
    #--- Si la carpeta ya existe, la borramos para empezar de cero (limpieza)
    if os.path.exists(CHROMA_PATH):
        print("Conextando a la base de datos existente y agregando el documento")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        )
    
        db.add_documents(chunks)  
    
    else:
    # Si no existeCreamos la DB y la guardamos en disco
        db = Chroma.from_documents(
            documents=chunks,
            embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
            persist_directory=CHROMA_PATH
        )
    


    print(f"✅ ¡Éxito! Base de datos guardada en la carpeta '{CHROMA_PATH}'.")
    print("   Ahora puedes apagar este script. La DB ya es persistente.")

if __name__ == "__main__":
        main()

