# IMPLIMENTATION-OF-RAG-PIPLINE

--------------------------------------------------------------------------------------------------------------

A. Data Ingestion

The process includes:

Chunking the Document: Breaking the text into smaller chunks to make retrieval more efficient.

Embedding the Chunks: Converting the chunks into vector representations using OpenAI embeddings.

Vector Database: Storing the embeddings in a vector database for fast retrieval during the question-answering phase.

--------------------------------------------------------------------------------------------------------------

LangChain code that handles the ingestion of the transcript:

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_loaders import TextLoader
from langchain.vectorstores import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import ChatOpenAI

# Step 1: Load transcript and create embeddings
openAIEmbeddings = OpenAIEmbeddings()
loader = TextLoader("./docs/intro-to-llms-karpathy.txt")

# Step 2: Create a vector database with embeddings
index = VectorstoreIndexCreator(embedding=openAIEmbeddings).from_loaders([loader])

# Step 3: Initialize the LLM and retrieval chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)

______________________________________________________________________________________________________________

B. RAG Process

steps:

Embed the Question: Convert the input question into an embedding vector using the same embedding model.

Retrieve Relevant Context: Query the vector database to find chunks of the transcript that are most relevant to the question.

Generate an Answer: Pass the retrieved context along with the question to the LLM, allowing it to generate an answer that is contextually accurate.

--------------------------------------------------------------------------------------------------------------
a sample query:

question = "What is retrieval augmented generation and how does it enhance the capabilities of large language models?"
result = qa_chain({"query": question})
print(result)

______________________________________________________________________________________________________________

Answering the Question List

The output should include:

The question.

The generated answer.

The retrieved context from the transcript.

--------------------------------------------------------------------------------------------------------------
Python script to handle this task:

import json

# Load the list of questions from the questions.json file
with open('questions.json', 'r') as f:
    questions = json.load(f)

json_results = []

# Process each question using the RAG pipeline
for question in questions:
    response = qa_chain({"query": question})
    
    # Store the question, answer, and context in the required format
    json_results.append({
        "question": question,
        "answer": response["result"],
        "contexts": [context.page_content for context in response["source_documents"]]
    })

# Save the results to a JSON file
with open('./my_rag_output.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(json_results, indent=4))
______________________________________________________________________________________________________________

 Evaluation with RAGAS

The next step is to evaluate the system using the RAGAS evaluation framework, which will output metrics like faithfulness, answer relevancy, context recall, and more.
--------------------------------------------------------------------------------------------------------------

To run the evaluation:

python eval.py ./my_rag_output.json

--------------------------------------------------------------------------------------------------------------

This will produce an output similar to:

{
    "results": {
        "faithfulness": 0.9149, 
        "answer_relevancy": 0.8295, 
        "context_recall": 0.8867, 
        "context_precision": 0.8478, 
        "answer_correctness": 0.7237
    }, 
    "ragas_score": 0.8404951600375368
}

______________________________________________________________________________________________________________

Optimization Techniques

experiment with different optimization techniques to improve the evaluation scores. Potential approaches include:

Tuning the chunk size of the transcript.

Adjusting the LLM parameters (e.g., temperature, max tokens).

Trying different embedding models or vector database configurations.

______________________________________________________________________________________________________________

Conclusion

This assignment demonstrates the practical implementation of a RAG pipeline for question-answering using the transcript of a talk by Andrej Karpathy. By following this process, you'll not only build a functioning RAG system but also apply an evaluation framework to gauge the performance and effectiveness of your solution.

______________________________________________________________________________________________________________

