
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

import re
import requests
import environ
import langchain
from langchain_ollama import ChatOllama
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
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
    """Main loop: Simplest approach, using 'create_sql_query_chain' and a basic prompt 
                template plus a list of examples of pairs of questions and their right
                sql queries.
    Args:
      ArgParse: a container for argument specifications
    Returns:
      None
    """
    model_name = get_model()

    print(60 * '-')
    print('[create_sql_query_chain + examples + prompt template]')
    print('[LangChain]', langchain.__version__)
    print('[LangModel]', model_name)
    print('===> Press Ctrl+C to exit! <===')

    # Initialize LLM
    llm = ChatOllama( model = model_name, temperature=0 )

    # Setup database
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://postgres:{env('DBPASS')}@localhost:5432/{env('DATABASE')}" # , schema='dbo'
    )

    # Write some guiding examples
    examples = [
        {"input": "List all artists.", "query": "SELECT * FROM artist;"},
        {
            "input": "Find all albums for the artist 'AC/DC'.",
            "query": "SELECT * FROM album WHERE artist_id = (SELECT artist_id FROM artist WHERE name = 'AC/DC');",
        },
        {
            "input": "List all tracks in the 'Rock' genre.",
            "query": "SELECT * FROM track WHERE genre_id = (SELECT genre_id FROM genre WHERE name = 'Rock');",
        },
        {
            "input": "Find the total duration of all tracks.",
            "query": "SELECT SUM(milliseconds) FROM track;",
        },
        {
            "input": "List all customers from Canada.",
            "query": "SELECT * FROM customer WHERE country = 'Canada';",
        },
        {
            "input": "How many tracks are there in the album 'Body Count'?",
            "query": "SELECT COUNT(*) FROM track WHERE album_id = (SELECT album_id FROM album WHERE title = 'Body Count');",
        },
        {
            "input": "Find the total number of invoices.",
            "query": "SELECT COUNT(*) FROM invoice;",
        },
        {
            "input": "List all tracks that are longer than 5 minutes.",
            "query": "SELECT * FROM track WHERE milliseconds > 300000;",
        },
        {
            "input": "Who are the top 5 customers by total purchase?",
            "query": "SELECT customer_id, SUM(total) AS TotalPurchase FROM invoice GROUP BY customer_id ORDER BY TotalPurchase DESC LIMIT 5;",
        },
        {
            "input": "What is the address of Jane Peacock?",
            "query": "SELECT address, city, state, postal_code, country FROM employee WHERE first_name = 'Jane' and last_name = 'Peacock' UNION SELECT address, city, state, postal_code, country FROM customer WHERE first_name = 'Jane' and last_name = 'Peacock';",
        },
        {
            "input": "What is the email of Daan Peeters?",
            "query": "SELECT email FROM customer WHERE first_name = 'Daan' and last_name = 'Peeters';",
        },
        {
            "input": "How many employees are there?",
            "query": "SELECT COUNT(*) FROM employee;",
        },
        {
            "input": "How many employees work as IT Staff?",
            "query": "SELECT COUNT(*) FROM employee WHERE title = 'IT STaff';",
        },
    ]

    # Create a FewShotPromptTemplate
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {query};"
    )

    template = """
        You are a postgresql expert. Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per postgresql. You can order the results to return the most informative data in the database.
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in single quotes (') to denote them as delimited identifiers.
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

        Only use the following tables: {table_info}.
    """
    prompt = FewShotPromptTemplate(
        example_prompt  = example_prompt,
        examples        = examples,
        prefix          = template,
        suffix          = "Question: {input}\nOutput:", 
        input_variables = ["input", "top_k", "table_info"],
    )

    def get_sql( query ):
        response = re.search("(SELECT.*);", query.replace("\n", " ")) #.group(1)
        if response == None:
            response = query
        else:
            response = response.group(1)

        return response

    write_query = create_sql_query_chain(llm, db, prompt)
    execute_query = QuerySQLDataBaseTool(db=db)
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )
    
    try:
        while True:
            print(60 * '-', '\n')
            query_txt = input( f'Enter your question (Ctrl+C to exit!): ' )
            print()
            raw_query = write_query.invoke({"question": query_txt})
            sql_query = get_sql(raw_query)
            # print(f'[SQL] {sql_query}')
            full_chain = (
                RunnablePassthrough.assign( result=itemgetter("query") | execute_query )
                | answer_prompt
                | llm
                | StrOutputParser()
            )
            for chunk in full_chain.stream( {"question": query_txt, "query": sql_query} ):
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
