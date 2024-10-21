
''' Chat with SQL database using Ollama 
endpoints and LangChain libraries.

Notes:
    Important -> Using Langchain V0.3!!!

Usage:    
    main.py [-h]

optional arguments:
    -h, --help  Show this help message and exit

    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-10-21
'''

import requests
import langchain
from langchain_ollama import ChatOllama

def get_model():
    """Shows a list of available LLMs and returns the user's selection .
    Args:
      List: A List of available Ollama models in host
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

def main_chat():
    """Main loop: Just chat with your LLM
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    model_name = get_model()

    print(60 * '-')
    print('Basic chat implementation')
    print('[LangChain]', langchain.__version__)
    print('[LangModel]', model_name)
    print('===> Press Ctrl+C to exit! <===')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )

    try:
        while True:
            print(60 * '-', '\n')
            query_txt = input( f'Enter your question (Ctrl+C to exit!): ' )
            print()
            for chunk in llm.stream( query_txt ):
                print(chunk.content, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print('Bye!')
    print()

    return

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another RAG Script".center(80))
    print(80 * '-')

    main_chat( )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
