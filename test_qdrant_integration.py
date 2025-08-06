#!/usr/bin/env python3
"""
Test script to verify Qdrant integration with OpenAI embeddings
for the PV prediction codebase indexing system.
"""

import os
from qdrant_client import QdrantClient
from openai import OpenAI

def test_qdrant_connection():
    """Test connection to Qdrant server"""
    try:
        client = QdrantClient("localhost", port=6333)
        collections = client.get_collections()
        print("SUCCESS: Qdrant connection successful!")
        print(f"Available collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"ERROR: Qdrant connection failed: {e}")
        return False

def test_openai_embeddings():
    """Test OpenAI embeddings API"""
    try:
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # Test embedding generation
        test_text = "def analyze_pv_data(): # Analyze photovoltaic power data"
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text
        )
        
        embedding = response.data[0].embedding
        print(f"SUCCESS: OpenAI embeddings working! Vector size: {len(embedding)}")
        return True, embedding
        
    except Exception as e:
        print(f"ERROR: OpenAI embeddings failed: {e}")
        return False, None

def test_vector_storage():
    """Test storing and retrieving vectors in Qdrant"""
    try:
        client = QdrantClient("localhost", port=6333)
        
        # Test with a sample code snippet from your PV project
        test_code = """
        class DataLoader:
            def __init__(self, dataset_host, dataset_file, all_cols):
                self.dataset_host = dataset_host
                self.dataset_file = dataset_file
                self.all_cols = all_cols
        """
        
        # Get embedding (mock for now if OpenAI fails)
        success, embedding = test_openai_embeddings()
        if not success:
            # Use mock embedding for testing
            embedding = [0.1] * 1536
            print("Using mock embedding for testing...")
        
        # Store test vector
        client.upsert(
            collection_name="python_pv_codebase",
            points=[{
                "id": 1,
                "vector": embedding,
                "payload": {
                    "code": test_code,
                    "file": "src/data_loader.py",
                    "class": "DataLoader",
                    "type": "class_definition"
                }
            }]
        )
        
        print("SUCCESS: Vector storage successful!")
        
        # Test search
        search_results = client.search(
            collection_name="python_pv_codebase",
            query_vector=embedding,
            limit=1
        )
        
        if search_results:
            print("SUCCESS: Vector search successful!")
            print(f"Found: {search_results[0].payload['class']} in {search_results[0].payload['file']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Vector storage/search failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Qdrant Integration for PV Prediction Codebase")
    print("=" * 60)
    
    # Test 1: Qdrant Connection
    print("\n1. Testing Qdrant Connection...")
    qdrant_ok = test_qdrant_connection()
    
    # Test 2: OpenAI Embeddings
    print("\n2. Testing OpenAI Embeddings...")
    result = test_openai_embeddings()
    if isinstance(result, tuple):
        openai_ok, _ = result
    else:
        openai_ok = result
    
    # Test 3: Vector Storage
    print("\n3. Testing Vector Storage & Search...")
    storage_ok = test_vector_storage()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Qdrant Connection: {'PASS' if qdrant_ok else 'FAIL'}")
    print(f"OpenAI Embeddings: {'PASS' if openai_ok else 'FAIL'}")
    print(f"Vector Storage: {'PASS' if storage_ok else 'FAIL'}")
    
    if all([qdrant_ok, storage_ok]):
        print("\nQdrant integration is ready for your PV prediction codebase!")
        print("\nNext steps:")
        print("- Index your Python files: src/data_loader.py, src/frequency_analyzer.py, etc.")
        print("- Index your research notebooks: src/jupyter_nb/*.ipynb")
        print("- Index your documentation: .kilocode/rules/memory-bank/*.md")
    else:
        print("\nSome tests failed. Please check the configuration.")

if __name__ == "__main__":
    main()