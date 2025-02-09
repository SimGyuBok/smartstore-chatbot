from app.database.vector_store import VectorStore

def main():
    print("Initializing Vector Store...")
    vector_store = VectorStore()
    
    print("Loading and indexing FAQ data...")
    vector_store.load_and_index_data()
    
    print("Database initialization completed!")

if __name__ == "__main__":
    main()