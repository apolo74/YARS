import os
import re
import json
import environ
# from operator import itemgetter

from dataclasses import dataclass

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# from langchain_core.runnables import RunnableLambda

from utils.load_config import LoadConfig

env = environ.Env()
environ.Env.read_env()

APPCFG = LoadConfig()

# ======================== Class: LLM Assistant ========================
@dataclass
class Assistant:
    """ Assistant Class"""
    #docs: list
    #pdf_path: str
    #index_path: str
    #k:int = DEFAULT_K
    with_database: bool = False

    def __post_init__(self):
        """
        Initializes an instance of the class with the given parameters.
        """
        self.model_name = 'llama3.2:1b'
        self.temperature = 0

        self.llm = ChatOllama( model = self.model_name )
        self.verb = False
        # self.emb = OllamaEmbeddings( model = "mxbai-embed-large" )
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

        # Setup database
        self.db = SQLDatabase.from_uri(
            f"postgresql+psycopg2://{env('DB_USER01')}:{env('DB_PASS01')}@localhost:{env('DB_PORT01')}/{env('DB_NAME01')}" # , schema='dbo'
        )
        # Opening JSON file
        abs_path = os.path.dirname(__file__) #<-- absolute dir the script is in

        with open(abs_path + '/sql_examples.json') as examples_file:
            self.all_examples = json.load(examples_file)
            self.sql_examples = self.all_examples['Chinook']

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

    def change_temperature(self, temp):
        """Transforms your PDF(s) into vector format and splits it(them) into chunks.
        Args:
            Float: A number between [0:1] for setting the temperature of the LLM
        Returns:
            Updated LLM
        """
        self.temperature = temp
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)

        print(f'[Temp ] {self.temperature}')

        return

    def change_model(self, model):
        """ Switches between Ollama LLMs
        Args:
            String: Name of the Ollama model to be used
        Returns:
            None
        """
        self.model_name = model
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)

        print(f'[Model] {self.llm.model}')

        return

    def change_mode(self, chat_mode):
        """ Switch between interaction modes
        Args:
            String: Mode flag between 'LLM' and 'SQL'
        Returns:
            None
        """
        self.with_database = False
        if chat_mode == 'SQL':
            self.with_database = True

        print(f'[Mode ] {chat_mode}')

        return
    
    def change_database(self, db_name):
        """ Updates values for database connection and sql examples
        Args:
            String:  Name of selected database
        Returns:
            None
        """
        if db_name == 'Chinook':
            DBUSER = env('DB_USER01')
            DBPASS = env('DB_PASS01')
            DBPORT = env('DB_PORT01')
            DBNAME = env('DB_NAME01')
            SUFFIX = 'postgresql+psycopg2'
        
        self.sql_examples = self.all_examples[db_name]
        db_uri = f"{SUFFIX}://{DBUSER}:{DBPASS}@localhost:{DBPORT}/{DBNAME}"

        self.db = SQLDatabase.from_uri(db_uri) # , schema='dbo'    )
        print(f'[ DB  ] {db_name}')

        return

    def change_verbose(self, flag_verbose):
        self.verb = True if flag_verbose else False

        return

    def get_sql_chain(self, db, query_txt):
        """Generates the SQL query to be executed in the final chain
        Args:
            SQLDatabase:  The postgres database to query
            String:       The question ask in natural language
        Returns:
            String:       The generated SQL query
        """
        # Create a FewShotPromptTemplate
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {query}"
        )
        # Create a semantic similarity example selector from the provided examples
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples        = self.sql_examples,
            embeddings      = OllamaEmbeddings(model=APPCFG.embedding_model),
            vectorstore_cls = FAISS,
            k               = 5,
            input_keys      = ["input"],
        )
        # Create a prompt template guided by examples
        few_shot_prompt = FewShotPromptTemplate(
            example_prompt  = example_prompt,
            example_selector= example_selector,
            prefix          = APPCFG.template_query, #template,
            suffix          = "Question: {input}\nOutput:", 
            input_variables = ["input", "top_k", "table_info"],
        )

        # Define the chain for generating the SQL query
        def extract_sql( raw_query ):
            # print(f'\n[SQL(raw)] {raw_query}')
            response = re.search("(SELECT.*);", raw_query.replace("\n", " "))
            if response == None:
                response = raw_query
            else:
                response = f'{response.group(1)}'
            # print(f'[SQL(out)] {response}')

            return response

        # Define the chain for generating the SQL query
        sql_chain = (
            RunnablePassthrough.assign(table_info=lambda _: db.get_table_info())
            | few_shot_prompt
            | self.llm
            | StrOutputParser()
        )

        # write_query = create_sql_query_chain(self.llm, db, prompt)
        # raw_query = write_query.invoke({"question": query_txt})
        raw_query = sql_chain.invoke({"input": query_txt, "top_k": 5})
        sql_query = extract_sql(raw_query)

        return sql_query
        
    def respond(self, message, chat_history):

        if self.with_database:
            sql_query = self.get_sql_chain(self.db, message)
            sql_run = self.db.run( sql_query )
            # execute_query = QuerySQLDataBaseTool(db=self.db)
            answer_prompt = PromptTemplate.from_template( APPCFG.template_answer )

            chain = (
                # RunnablePassthrough.assign( result=RunnableLambda(lambda x: self.db.run(sql_query)))
                # RunnablePassthrough.assign( result=itemgetter("query") | execute_query )
                answer_prompt
                | self.llm
                | StrOutputParser()
            )

            output = f"[SQL] {sql_query};\n[Run] {sql_run};\n[LLM] " if self.verb else ''
            for chunk in chain.stream(  {"question": message, "query": sql_query, "result": sql_run}  ):
                output = output + chunk
                yield output
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", APPCFG.template_chat),
                ("human", message)
            ])
            chain = prompt | self.llm
            output = ""
            for chunk in chain.stream( {'question': message} ):
                output = output + chunk.content
                yield output

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": output})
        
        return "", chat_history
    


