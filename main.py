''' Chat with SQL database using Ollama 
endpoints and LangChain/LangGraph libraries.

Notes:
    - Important -> Using Langchain V0.3.3!!!
    - Important -> Works well with llama3.2:3b

Usage:    
    main.py [-h] [-s] examples

    positional arguments:
        examples    Path to a JSON file with SQL examples.

    options:
        -h, --help  show this help message and exit
        -s, --sql   Show the generated SQL query!

    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-11-06
'''

import re
import requests
import json
import environ
import argparse

import langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from utils.load_config import LoadConfig

env = environ.Env()
environ.Env.read_env('utils/.env')

APPCFG = LoadConfig()

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

def get_sql_query(llm, db, query_txt, examples_path):
    """Generates the SQL query to be executed in the final chain
    Args:
      LLM:          The language model selected from the Ollama server
      SQLDatabase:  The postgres database to query
      String:       The question ask in natural language
      Dict:         A dictionary with example questions and their SQL queries
    Returns:
      String:       The generated SQL query
    """
    # Opening JSON file
    with open( examples_path ) as examples_file:
        all_examples = json.load(examples_file)
        sql_examples = all_examples['Chinook']
    
    # Create a FewShotPromptTemplate
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {query};"
    )

    # Create a semantic similarity example selector from the provided examples
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples        = sql_examples,
        embeddings      = OllamaEmbeddings(model=APPCFG.embedding_model),
        vectorstore_cls = FAISS,
        k               = 5,
        input_keys      = ["input"],
    )

    # Create a prompt template guided by examples
    few_shot_prompt = FewShotPromptTemplate(
        example_prompt  = example_prompt,
        example_selector= example_selector,
        # examples        = examples,
        prefix          = APPCFG.template_query,
        suffix          = "Question: {input}\nOutput:", 
        input_variables = ["input", "top_k", "table_info"],
    )

    # Define the chain for generating the SQL query
    sql_chain = (
        RunnablePassthrough.assign(table_info=lambda _: db.get_table_info())
        | few_shot_prompt
        | llm
        | StrOutputParser()
    )

    # Function for defining a proper SQL query
    def extract_sql( raw_query ):
        response = re.search("(SELECT.*);", raw_query.replace("\n", " "))
        if response == None:
            response = raw_query
        else:
            response = f'{response.group(1)};'

        return response

    raw_query = sql_chain.invoke({"input": query_txt, "top_k": 5})
    sql_query = extract_sql(raw_query)

    return sql_query
    
def get_agent_answer(llm, tools, sql_query):
    # Create a react agent
    system_prompt = """You are a helpful postgres agent named Burrito.
        Your job is to respond in natural language using the results of executing the SQL query. 
        Personal names should be enclosed between double quotes.
    """
    agent = create_react_agent(llm, tools, state_modifier=system_prompt)

    events = agent.stream({"messages": [("user", sql_query)]}, stream_mode='messages')
    for msg, metadata in events:
        # print(f'[msg] {msg} \n[mData] {metadata}\n')
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "agent"
        ):
            print(msg.content, end="", flush=True)
    print()

    return

def main_loop( args ):
    """Main loop: Using 'create_react_agent' and a basic prompt 
                template plus a list of examples of pairs of questions 
                and their right sql queries.
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    # Ask for the LLM to use
    model_name = get_model()
    
    # Read the path to the examples file.
    examples_path = args.examples

    # Summarize main parameters
    print(60 * '-')
    print('[SQL examples -> SQL generating chain -> Response chain]')
    print(f'{'[LangChain Version]':.<30} {langchain.__version__}')
    print(f'{'[Language Model]':.<30} {model_name}')
    print(f'{'[Embeddings Model]':.<30} {APPCFG.embedding_model}')
    print(f'{'[SQL examples]':.<30} {examples_path}')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )

    # Setup database
    uri_conn = f"postgresql+psycopg2://{env('DB_USER')}:{env('DB_PASS')}@localhost:{env('DB_PORT')}/{env('DB_NAME')}"
    db = SQLDatabase.from_uri( uri_conn )
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = sql_toolkit.get_tools()
    
    try:
        while True:
            print(60 * '-', '\n')
            query_txt = input( f'Enter your question (Ctrl+C to exit!): ' )
            print()
            # Get the SQL query
            sql_query = get_sql_query(llm, db, query_txt, examples_path)
            if args.sql: print(f'[SQL] {sql_query}')
            get_agent_answer(llm, tools, sql_query)
    except KeyboardInterrupt:
        print('Bye!')
    print()
    
    return

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    parser = argparse.ArgumentParser(description='Chat with your documents')
    parser.add_argument('examples', #required=True, 
        help='Path to a JSON file with SQL examples.')
    parser.add_argument('-s', '--sql', action='store_true', default=False, 
        help='Show the generated SQL query!')

    args = parser.parse_args()
    main_loop( args )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
