from langchain_core.documents import Document
from typing import List
from pprint import pprint
import textwrap
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader
)
from langchain_text_splitters import (RecursiveCharacterTextSplitter)

#region PDF Loaders
'''
### PypdfLoader
try:
    pdf_loader = PyPDFLoader("mine_udemy/data/pdf/attention.pdf")
    pypdf_docs=pdf_loader.load()
    print(pypdf_docs)
    print(f"  Loaded {len(pypdf_docs)} pages")
    print(f"  Page 1 content: {pypdf_docs[0].page_content[:100]}...")
    print(f"  Metadata: {pypdf_docs[0].metadata}")

except Exception as e:
    print(f"Error : {e}")


# Method 2: PyMuPDFLoader (Fast and accurate)
print("\n3️⃣ PyMuPDFLoader")
try:
    pymupdf_loader = PyMuPDFLoader("mine_udemy/data/pdf/attention.pdf")
    pymupdf_docs = pymupdf_loader.load()
    
    print(f"  Loaded {len(pymupdf_docs)} pages")
    print(f"  Includes detailed metadata")
    print(pymupdf_docs)
except Exception as e:
    print(f"  Error: {e}")
'''
#endregion

class SmartPdfProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter (
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            separators=[" "]
        )

    def process_pdf(self, pdf_path:str) -> list[Document]:
        loader = PyPDFLoader(pdf_path)
        pages=loader.load()

        ## Process each page
        processed_chunks=[]

        for pagenum,page in enumerate(pages):
            cleaned_text = self._clean_text(page.page_content)
            chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[
                    {
                        **page.metadata,
                        "page": pagenum+1,
                        "total_pages": len(pages),
                        "chunk_method": "smart_pdf_processot",
                        "char_count": len(cleaned_text)
                    }
                ]
            )

            processed_chunks.extend(chunks)

            return processed_chunks


    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common PDF extraction issues
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        
        return text


if __name__ == "__main__":
    # Initialize the processor
    processor = SmartPdfProcessor(chunk_size=1000, chunk_overlap=100)
    
    # Process the PDF
    pdf_path = "mine_udemy/data/pdf/attention.pdf"
    
    try:
        chunks = processor.process_pdf(pdf_path)
        print(f"Successfully processed PDF into {len(chunks)} chunks")
        
        # Display sample chunk (pretty-printed)
        if chunks:

            first = chunks[0]
            print("\nSample chunk:")
            print("-" * 80)
            print("Content:")
            print(textwrap.fill(first.page_content.strip(), width=100))
            print("\nMetadata:")
            pprint(first.metadata, sort_dicts=True, indent=2, width=100)
    except Exception as e:
        print(f"Error processing PDF: {e}")