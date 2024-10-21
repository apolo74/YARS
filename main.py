
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
import environ
import langchain
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter

env = environ.Env()
environ.Env.read_env()

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

def main_sql():
    """Main loop: Simplest approach, using 'create_sql_query_chain'
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    model_name = get_model()

    print(60 * '-')
    print('[create_sql_query_chain]')
    print('[LangChain]', langchain.__version__)
    print('[LangModel]', model_name)
    print('===> Press Ctrl+C to exit! <===')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )

    # Setup database
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}" # , schema='dbo'
    )

    write_query = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )
    
    chain = (
        RunnablePassthrough.assign( query=write_query ).assign( 
            result=itemgetter("query") | execute_query )
        | answer_prompt
        | llm 
        | StrOutputParser()
    )

    try:
        while True:
            print(60 * '-', '\n')
            query_txt = input( f'Enter your question (Ctrl+C to exit!): ' )
            print()
            for chunk in chain.stream( {"question": query_txt} ):
                print(chunk, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print('Bye!')
    print()
    
    return
if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another RAG Script".center(80))
    print(80 * '-')

    main_sql( )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
