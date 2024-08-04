import gradio as gr
import arxiv
import fitz

# from langchain import LLMChain, PromptTemplate
# from langchain.llms import Groq
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up Groq API (replace with your actual API key)
from groq import Groq

groq_api_key = "<Your-api-key>"
gclient = Groq(api_key=groq_api_key)

# Set up embedding model and vector store
"""embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_texts([], embeddings)"""


client = arxiv.Client()


def search_paper(query):
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    print("Searching Results")
    results = client.results(search)
    print("FOund Results")
    for result in results:
        return result

prompt = """
    Taking the following context delimited by triple backquotes into consideration:

    ```{context}```

    Write a concise summary of the following text delimited by triple backquotes. You must keep the core mechanisms and terms introdcued here.

    ```{text}```

    CONCISE SUMMARY:
"""

import time


def chatbot(input_text, history):
    # Implement chatbot logic here
    # This should handle paper search, question answering, and summarization
    print("Starting Search")
    results = search_paper(input_text)
    # print(results[0])
    print("Read Results")
    if not results:
        return "No such Paper Found!!"

    results.download_pdf(filename="curpap.pdf")
    doc = fitz.open("curpap.pdf")
    summ = []
    i = 0

    print("Reading")

    for page in doc:
        print("paged")
        if i == 0:
            p2 = prompt.format(context = "No context", text= page.get_text())
        else:
            p2 = prompt.format(context = summ[-1], text = page.get_text())

        print(i)

        chat_completion = gclient.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"This is the page of a paper titled {results.title}. f{p2}",
                }
            ],
            model="llama3-8b-8192",
        )
        summ.append(chat_completion.choices[0].message.content)

        time.sleep(3)
        i += 1

    final_prompt_template = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key mechnisms of the text.

    ```{text}```

    BULLET POINT SUMMARY:
"""

    with open("Latestout.txt","w") as f:
        f.write("\n".join(summ))
    chatoutput = gclient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": final_prompt_template.format(text = "\n".join(summ)),
            }
        ],
        model="llama3-8b-8192",
    )

    return f"{results.title}\n{chatoutput.choices[0].message.content}"


iface = gr.ChatInterface(
    chatbot,
    title="arXiv Paper Assistant",
    description="Search for papers, ask questions, and get summaries.",
)

iface.launch()
