import os
import warnings
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# 1. Silenciamos el warning de urllib3/requests para ver la consola limpia
warnings.filterwarnings("ignore")

# 2. Cargamos las variables del archivo .env
load_dotenv()

def test_connection():
    print("--- 🔐 Iniciando conexión segura a Elasticsearch ---")
    
    # 3. Recuperamos los datos del entorno
    # 'elastic' es el usuario por defecto de Elasticsearch
    username = "elastic"
    password = os.getenv("ELASTIC_PASSWORD")
    
    if not password:
        print("❌ ERROR: No se encontró 'ELASTIC_PASSWORD' en el archivo .env")
        return

    # 4. Configuramos el cliente con 'basic_auth'
    client = Elasticsearch(
        "http://localhost:9200",
        basic_auth=(username, password)
    )
    
    try:
        # Forzamos la petición de info
        info = client.info()
        
        print("✅ ¡CONEXIÓN EXITOSA!")
        print(f"👤 Usuario: {username}")
        print(f"📦 Cluster: {info['cluster_name']}")
        print(f"🏷️ Versión: {info['version']['number']}")
            
    except Exception as error:
        print("\n❌ ERROR DE AUTENTICACIÓN O CONEXIÓN:")
        print(f"Tipo: {type(error).__name__}")
        print(f"Detalles: {error}")

if __name__ == "__main__":
    test_connection()