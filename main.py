
''' Chat with SQL database using Ollama 
endpoints and LangChain libraries.

Notes:
  Important -> Using Langchain V0.3!!!

Usage:    
        main.py [-h] -d DOCUMENT

optional arguments:
    -h, --help                          Show this help message and exit

required arguments:
    -d DOCUMENT, --document DOCUMENT    Path to document or directory.
    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-10-14
'''

import os
import argparse
import getpass
import requests

# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentType

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

import environ
env = environ.Env()
environ.Env.read_env()

from datetime import datetime

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

    return model_name

def main_loop(  ):
    """Main loop where all magic happens!
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    local_models = requests.get('http://localhost:11434/api/tags').json()

    model_name = get_model( local_models['models'] )

    print(60 * '-')
    print('Working with model:', model_name)
    print('===> Press Ctrl+C to exit! <===')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )

    # Setup database
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}" # , schema='dbo'
    )

    system = """You are an Postgres expert agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    Wrap each column name in single quotes (') to denote them as delimited identifiers. 
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. 
    Also, pay attention to which column is in which table.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    You have access to the following tables: {table_names}

    If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
    Do not try to guess at the proper name - use this function to find similar ones.""".format(
        table_names=db.get_usable_table_names()
    )

    system_message = SystemMessage(content=system)

    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = sql_toolkit.get_tools()

    agent = create_react_agent(llm, tools, state_modifier=system_message)
    ''' 
    inputs = {"messages": [HumanMessage(content="How many albums does alis in chain have?")]}
    for s in agent.stream(inputs, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    '''
    try:
        while True:
            print(60 * '-', '\n')
            query_txt = input( f'Enter your prompt (Ctrl+C to exit!): ' )
            inputs = {"messages": [HumanMessage(content=query_txt)]}
            print()
            ix = 0
            for results in agent.stream( inputs, stream_mode="values" ):
                message = results["messages"][-1] # .pretty_print()
                print(f'{ix:2d}: {message.content}') # 
                ix = ix + 1
    except KeyboardInterrupt:
        print('Bye!')
    print()

    return

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another RAG Script".center(80))
    print(80 * '-')

    # parser = argparse.ArgumentParser(description='Chat with your documents')
    # parser.add_argument('-d', '--document', required=True, help='Path to document or directory.')

    # args = parser.parse_args()
    main_loop( )#args )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
