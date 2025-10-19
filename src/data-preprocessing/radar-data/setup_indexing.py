import requests
import json

try:
    # Test if Qdrant is running
    response = requests.get("http://localhost:6333/collections")
    
    # Create collection using HTTP API
    collection_config = {
        "vectors": {
            "size": 1536,
            "distance": "Cosine"
        }
    }
    
    response = requests.put(
        "http://localhost:6333/collections/python_pv_codebase",
        json=collection_config
    )
    
    if response.status_code in [200, 201]:
        print("Qdrant collection 'python_pv_codebase' created successfully!")
    else:
        print(f"Error creating collection: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to Qdrant server")
    print("Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
except Exception as e:
    print(f"Error: {e}")