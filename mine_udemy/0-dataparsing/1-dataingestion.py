import os
import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

#region LOAD Langchain Document
'''
doc = Document(
    page_content="Concerns about an AI bubble have been simmering for at least a year and a half. Nvidia's incredibly strong earnings this past week tried to put those fears to bed. It may not have been enough.low world",
    metadata = {
        "source": "example.txt",
        "page": 1,
        "author": "Sai Pavan",
        "date_created": "11/23/2025",
        "custom_field":"any value"
    }
)
'''
# endregion

#region create txt files
'''
os.makedirs("data/text_files", exist_ok=True)

sample_texts = {
    "data/text_files/python_intro.txt": """
# Introduction to Python

Python is a versatile, beginner-friendly programming language known for its clear and readable syntax. It was created by Guido van Rossum and first released in 1991, and it has since become one of the most popular programming languages in the world.

**Why Python is Popular**

Python's popularity stems from several key features. Its syntax is intuitive and closely resembles natural language, making it easier for beginners to learn compared to languages like C++ or Java. It's also incredibly versatile—you can use Python for web development, data analysis, artificial intelligence, scientific computing, automation, and much more.

**Key Characteristics**

Python is dynamically typed, meaning you don't need to declare variable types explicitly. It's also interpreted, so you can write code and run it immediately without compiling. The language emphasizes code readability through its use of indentation to define code blocks, which forces developers to write clean, organized code.

**What You Can Build With Python**

Python powers many modern applications and tools. Data scientists use libraries like NumPy, Pandas, and Matplotlib for analysis and visualization. Machine learning engineers rely on frameworks like TensorFlow and scikit-learn. Web developers use frameworks like Django and Flask to build web applications. Automation scripts help with repetitive tasks, and Python is also used in scientific research, finance, and game development.

**Getting Started**

To begin learning Python, you'll need to install it from python.org. Many beginners start by learning basic concepts like variables, data types (strings, numbers, lists), conditionals, loops, and functions. Once you're comfortable with these fundamentals, you can explore libraries and frameworks relevant to your interests.

Python's extensive documentation and large, supportive community make it an excellent choice for both beginners and experienced programmers.
""",
"data/text_files/machine_learning.txt": """
    # Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every scenario. Instead of following pre-written rules, machine learning systems improve their performance through experience and exposure to data.

**How Machine Learning Works**

In traditional programming, you write explicit instructions for a computer to follow. Machine learning inverts this approach—you provide data and desired outcomes, and the algorithm learns patterns from that data to make future predictions. For example, rather than coding rules to identify spam emails, a machine learning model learns what spam looks like by analyzing thousands of examples and can then classify new emails automatically.

**Types of Machine Learning**

Machine learning generally falls into three categories. Supervised learning uses labeled data (where correct answers are known) to train models for tasks like prediction and classification. Unsupervised learning finds patterns in unlabeled data without predefined answers, useful for clustering and data exploration. Reinforcement learning trains agents to make decisions by rewarding desired behaviors, commonly used in robotics and game AI.

**Real-World Applications**

Machine learning powers many technologies you use daily. Recommendation systems suggest movies or products based on your preferences. Computer vision enables facial recognition and medical image analysis. Natural language processing powers chatbots and language translation. Credit scoring, fraud detection, and autonomous vehicles are other common applications.

**Key Concepts**

Training involves feeding data to an algorithm so it learns patterns. Validation checks if the model performs well on new, unseen data. Features are the input variables your model uses to make predictions. The goal is to build models that generalize well—they perform accurately not just on training data but on real-world data they've never seen before.

**Getting Started**

If you're interested in machine learning, Python is an excellent choice due to libraries like scikit-learn, TensorFlow, and PyTorch that simplify development. Most paths begin with understanding statistics, linear algebra, and data manipulation before diving into specific algorithms and deep learning concepts.
"""
}

for filepath, content in sample_texts.items():
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write(content)

print("My sample file got created")

'''
# endregion

#region load text files using Langchain Text loader
'''
loader = TextLoader("data/text_files/python_intro.txt", encoding="utf-8")

documents = loader.load()

print(f"Type: {type(documents)}")
print(f"\nNumber of documents: {len(documents)}")
print("\n" + "="*80)

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print(f"{'='*80}")
    print(f"\nMetadata: {doc.metadata}")
    print(f"\nContent Preview (first 500 chars):")
    print(f"{'-'*80}")
    print(doc.page_content[:5000])
    print(f"\n{'='*80}")
'''

# endregion

dir_loader = DirectoryLoader(
    "mine_udemy/data/text_files", 
    glob = "**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding':"utf-8"},
    show_progress=True
)

documents = dir_loader.load()

for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print(f"{'='*80}")
    print(f"\nMetadata: {doc.metadata}")
    print(f"\nContent Preview (first 5000 chars):")
    print(f"{'-'*80}")
    print(doc.page_content[:5000])
    print(f"\n{'='*80}")

#region Splitter strategies
# 1. character based splitting

text = documents[0].page_content

print("1. CHARACTER TEXT SPLITTING")
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 200,
    chunk_overlap = 20,
    length_function = len
)
char_chunks = char_splitter.split_text(text)

print(f"Total Chunks: {len(char_chunks)}")

print("\nSample Chunks:")
for i, chunk in enumerate(char_chunks[:3]):  # Show first 3 chunks as samples
    print(f"Chunk {i+1}:")
    print(chunk)
    print(f"{'-'*80}")

# 2. Recursive character text splitter
rec_text_splitter = RecursiveCharacterTextSplitter(
    separators=[" "],
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)

recursive_chunks = rec_text_splitter.split_text(text)
print(f"Total Chunks: {len(recursive_chunks)}")

print("\nSample Chunks:")
for i, chunk in enumerate(recursive_chunks[:3]):  # Show first 3 chunks as samples
    print(f"Chunk {i+1}:")
    print(chunk)
    print(f"{'-'*80}")

#endregion

