from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

'''
## MEthod1: Using Docx2txtLoader
print("1️⃣ Using Docx2txtLoader")
try:
    doc_loader = Docx2txtLoader("mine_udemy/data/word_files/proposal.docx")
    docs = doc_loader.load()
    for doc in docs:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)
except Exception as e:
    print(f"Error: {e}")

'''

## MEthod2
print("\n2️⃣ Using UnstructuredWordDocumentLoader")

try:
    unstructured_loader=UnstructuredWordDocumentLoader("mine_udemy/data/word_files/proposal.docx",mode="elements")
    unstructured_docs=unstructured_loader.load()
    
    print(f"✅ Loaded {len(unstructured_docs)} elements")
    for i, doc in enumerate(unstructured_docs[:3]):
        print(f"\nElement {i+1}:")
        print(f"Type: {doc.metadata.get('category', 'unknown')}")
        print(f"Content: {doc.page_content[:100]}...")


except Exception as e:
    print(e) 
