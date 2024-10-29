import getpass
import os
import time
from pinecone import Pinecone, ServerlessSpec

def setup_pinecone_environment():
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    return os.environ.get("PINECONE_API_KEY")

def create_index(pc, index_name):
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    spec = ServerlessSpec(
        cloud="aws",
        region="eu-central-1"  # Changed to eu-central-1 for better proximity to Portugal
    )

    if index_name not in existing_indexes:
        logger.info(f"Creating index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=4096,  # dimensionality of ada 002
                metric='cosine',
                spec=spec
            )
            
            start_time = time.time()
            while not pc.describe_index(index_name).status['ready']:
                elapsed_time = time.time() - start_time
                if elapsed_time > 300:  # Timeout after 5 minutes
                    raise TimeoutError(f"Index creation timed out after {elapsed_time:.2f} seconds.")
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return None
    
    
    print(pc.Index(index_name).describe_index_stats())
    
    return pc.Index(index_name)

def main():
    pinecone_api_key = setup_pinecone_environment()
    if not pinecone_api_key:
        raise ValueError("Pinecone API key não encontrada.")
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    index_name = "my-pharma-index"
    index = create_index(pc, index_name)

if __name__ == "__main__":
    main()
