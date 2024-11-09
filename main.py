
''' Chat with PDF documents using Ollama 
endpoints and LangChain libraries.

Notes:
  Choose between pre-trained LLM models

Usage:    
        main.py [-h] -d DOCUMENT

optional arguments:
    -h, --help                          Show this help message and exit

required arguments:
    -d DOCUMENT, --document DOCUMENT    Path to document or directory.
    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-11-09
'''

import os
import argparse
import requests

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_model():
    """Shows a list of available Ollama LLMs and returns the user's selection .
    Args:
      None
    Returns:
      String: Name of the selected chat model.
    """
    local_models = requests.get('http://localhost:11434/api/tags').json()

    models = local_models['models']

    models_list = []
    for ix, model in enumerate(models):
        models_list.append(model['name'])
    
    models_dict = {}
    for ix, model in enumerate(sorted( models_list )):
        models_dict[ix] = model
        print(f'{ix:>3}: {model}')

    model_ix = int( input( f'Choose your Model (0-{ix}): ' ) )
    if model_ix < len(models_dict): 
        ix_exist = True
        model_name = models_dict[model_ix]

    return model_name

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

def main_loop( args ):
    """Main loop where all magic happens!
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    # Ask for the LLM to use
    model_name = get_model()

    # Summarize main parameters
    print(60 * '-')
    print('[SQL examples -> SQL generating chain -> Response chain]')
    print(f'{'[Language Model]':.<30} {model_name}')
    print(f'{'[SQL examples]':.<30} {args.document}')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )
    # Initialize Embeddings
    emb = OllamaEmbeddings(model="mxbai-embed-large")
    # Read the documents
    docs = get_pdf_langchain( args.document )

    vectorstore = FAISS.from_documents( docs, emb )
    retriever = vectorstore.as_retriever()
       
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Answer solely based on the following context:\n<Documents>\n{context}\n</Documents>",),
         ("user", "{question}"),]
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        while True:
            print(60 * '-', '\n')
            print( 'Enter your question (Ctrl+C to exit!) ' )
            query_txt = input( '[Question] ' )
            print(f'[ Answer ] ', end='', flush=True)
            for chunk in chain.stream( query_txt ):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print('Bye!')
    print()

    return

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    parser = argparse.ArgumentParser(description='Chat with your documents')
    parser.add_argument('-d', '--document', required=True, help='Path to document or directory.')

    args = parser.parse_args()
    main_loop( args )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
