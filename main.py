
''' Chat with PDF documents using Ollama endpoints and LangChain
libraries. A GUI (Graphic User Interface) is provided and implemented
using the Gradio library.

Usage:    
        main.py [-h] -d DOCUMENT

optional arguments:
    -h, --help                          Show this help message and exit
    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-07-09
'''

import os
import requests

import gradio as gr
from dataclasses import dataclass

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore
# from langchain.docstore.document import Document
from faiss import IndexFlatL2

css = """
.container {
    height: 15vh;
}
#chatbot {
    flex-grow: 1 !important;
    overflow: auto !important;
}
"""
#col { height: calc(100vh - 112px - 16px) !important; }

# ======================== Class: LLM Assistant ========================
@dataclass
class Assistant:
    """ Assistant Class"""
    #docs: list
    #pdf_path: str
    #index_path: str
    #k:int = DEFAULT_K
    with_context: bool = False

    def __post_init__(self):
        """
        Initializes an instance of the class with the given parameters.
        """
        self.model_name = 'phi3:latest'

        self.llm = ChatOllama( model = self.model_name )
        self.emb = OllamaEmbeddings( model = "mxbai-embed-large" )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        dimensions: int = len(self.emb.embed_query("dummy"))
        self.index = FAISS(embedding_function=self.emb, 
                           index=IndexFlatL2(dimensions), 
                           docstore=InMemoryByteStore(), 
                           index_to_docstore_id={}
                        )

    def get_models_list( self, models ):
        """Shows a list of available LLMs and returns the user's selection .
        Args:
            List: A List of available Models from Ollama's server list
        Returns:
            List: An alphabetically ordered list with Ollama's available models.
        """
        models_list = []
        for ix, model in enumerate(models):
            models_list.append(model['name'])

        return sorted( models_list )

    def change_model(self, model):
        self.model_name = model
        self.llm = ChatOllama(model=self.model_name)

        print('Working with: ', self.llm)

        return

    def ingest_pdf( self, file ):
        """Transforms your PDF(s) into vector format and splits it(them) into chunks.
        Args:
            String: Path to a file or a directory
        Returns:
            List: A list with chunked Documents
        """
        print(f'Loading {file} ...', end='')
        loader = PyMuPDFLoader(file)
        print(' done!')
        
        pages = loader.load_and_split( self.text_splitter )

        self.index = FAISS.from_documents( pages, self.emb )

        self.with_context = True

        return

    def clear_pdf(self):
        self.with_context = False

    def respond(self, question, history):

        if self.with_context:

            PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum. Keep the answer as concise as possible.
            {context}
            Question: {question}
            Helpful Answer:"""
            prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            
            '''
            prompt = ChatPromptTemplate.from_messages([
                ("system", "If you don't know the answer, just say that you don't know. Keep the answer as concise as possible. Answer based only on the following context:\n<Documents>\n{context}\n</Documents>",),
                ("human", question)
            ])
            '''
            chain = (
                {
                    "context": self.index.as_retriever(), 
                    "question": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            output = ""
            for chunk in chain.stream( question ):
                output = output + chunk
                yield output
        else:
            #    ("system", "You are a technical assistant. Answer with consice but informative formal tone to the given question! If you don't know the answer, just say that you don't know. "),
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a friendly assistant called Nobody. Answer with a casual tone to the given question! If you don't know the answer, just say that you don't know. "),
                ("human", question)
            ])
            chain = prompt | self.llm
            output = ""
            for chunk in chain.stream( {'text': prompt} ):
                output = output + chunk.content
                yield output



def get_model( models ):
    """Shows a list of available LLMs and returns the user's selection .
    Args:
      List: A List of available Ollama models in host
    Returns:
      String: Name of the selected chat model.
    """
    models_list = []
    for ix, model in enumerate(models):
        models_list.append(model['name'])
    
    models_dict = {}
    for ix, model in enumerate(sorted( models_list )):
        models_dict[ix] = model
        print(f'{ix:>3}: {model}')

    model_ix = int( input( f'Choose your LLM: ' ) )
    if model_ix < len(models_dict): 
        ix_exist = True
        model_name = models_dict[model_ix]

    return models_list

def get_pdf_langchain( path ):
    """Transforms your PDF(s) into vector format and splits it(them) into chunks.
    Args:
      String: Path to a file or a directory
    Returns:
      List: A list with Documents
    """
    if os.path.isdir(path):
        print('Argument is a folder!')
        #print(len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))
        documents = []
        for ix, file in enumerate(os.listdir(path)):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(path, file)
                print(f'Loading {file} ...', end='')
                loader = PyMuPDFLoader(pdf_path)
                print(' done!')
                pages = loader.load_and_split()
                documents.extend( pages )
    else:
        print('Argument is a file!')
        print(f'Loading {path} ...', end='')
        loader = PyMuPDFLoader(path)
        print(' done!')
        print('Splitting in chunks:', end='')
        pages = loader.load_and_split()
        print(' ... done!')
        documents = pages
    print('Documents: ', len(documents))

    return pages

def main_loop():
    """Main loop where all magic happens!
    Args:
      None
    Returns:
      None
    """
    assistant = Assistant()    
    local_models = requests.get('http://localhost:11434/api/tags').json()
    models = assistant.get_models_list(local_models['models'])

    # Define UI
    PLACE_HOLDER = "Ask me something about your docs!"
    with gr.Blocks(css=css) as all_blocks:
        with gr.Column(elem_classes=["container"]):
            gr.Markdown(
                """
                # Boris's First Chatbot!
                Upload a PDF and start asking questions about it, enjoy!
                """)    
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    with gr.Row():
                        dd_model = gr.Dropdown(models, value=assistant.model_name, label="Models", interactive=True)
                        dd_embedder = gr.Dropdown(["nomic", "phi3"], label="Embedders")
                    with gr.Row():
                        tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'])
                        tb_url = gr.Textbox(label="URL")
                with gr.Column(scale=5):
                    gr.ChatInterface(
                        fn=assistant.respond,
                        chatbot=gr.Chatbot(elem_id="chatbot", render=False, bubble_full_width=False),
                        clear_btn=None,
                        retry_btn=None,
                        undo_btn=None,
                        submit_btn="Submit",
                        # fill_height=True,
                        # theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"),
                    )
            dd_model.change(assistant.change_model, dd_model)
            tb_file.upload(assistant.ingest_pdf, tb_file, show_progress='full')
            tb_file.clear(assistant.clear_pdf)
    
    return all_blocks.launch()

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another RAG Script".center(80))
    print(80 * '-')

    main_loop()

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
