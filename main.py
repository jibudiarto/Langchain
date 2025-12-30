from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate 
from vector import retriever #import the retriever from vector.py

# Model Configuration
model = OllamaLLM(model="llama3.2", 
                  temperature=0.1,
                  max_tokens=1024)

# Pre-config template for LLM
template = """
You are an expert in book recommendations.
Here is a list of books froom GoodReads website that might be relevant to the user's question: {book_list}.
Provide a detailed answer to the following question: {question}.
Please include average rating, date published, and book_id everytime you recommends.
Also describe the authors of the book and what is their famously known for.
"""

prompt = ChatPromptTemplate.from_template(template)
chain  = prompt | model

while True:
    print("\n=============================================\n")
    question = input("Describe what kind of book you're looking for : (type 'q' to quit) ")
    print("\n=============================================\n")
    if question == "q":
        break

    book_list = retriever.invoke(question) # Retrieve the book list vector from ChromaDB
    result = chain.invoke({"book_list": book_list, "question": question})
    print(result)




