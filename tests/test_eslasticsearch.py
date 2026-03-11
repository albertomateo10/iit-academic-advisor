from elasticsearch import Elasticsearch
import warnings

warnings.filterwarnings("ignore")

def test_connection():
    """
    Tests the connection to the local Elasticsearch Docker container.
    """
    print("Initiating connection to Elasticsearch...")
    
    # We use localhost since we know your browser reached it successfully there
    client = Elasticsearch("http://localhost:9200")
    
    try:
        # We skip the ping() and force it to ask for the database info directly.
        # If there is a connection issue, this line will violently crash and tell us why.
        info = client.info()
        
        print("✅ Connection successful! Python is communicating with Docker.")
        print(f"📦 Cluster name: {info['cluster_name']}")
        print(f"🏷️ Elasticsearch version: {info['version']['number']}")
            
    except Exception as error:
        # This will catch the EXACT technical reason it's failing
        print("\n❌ FATAL CONNECTION ERROR:")
        print(f"Type: {type(error).__name__}")
        print(f"Details: {error}")

if __name__ == "__main__":
    test_connection()