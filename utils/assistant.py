import os
import re
import json
import environ

from openai import OpenAI

from dataclasses import dataclass

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryByteStore
from faiss import IndexFlatL2

from utils.load_config import LoadConfig

env = environ.Env()
environ.Env.read_env()

APPCFG = LoadConfig()

# ======================== Class: LLM Assistant ========================
@dataclass
class Assistant:
    """ Assistant Class"""
    with_database: bool = False
    with_images: bool = False
    with_context: bool = False

    def __post_init__(self):
        """
        Initializes an instance of the class with the given parameters.
        """
        abs_path = os.path.dirname(__file__) #<-- absolute dir the script is in
        self.verbose = False                # Outputs intermediate steps in the SQL mode

        # Setup chat parameters
        self.llm_model_name = 'llama3.2:1b'
        self.emb_model_name = 'mxbai-embed-large'
        self.temperature = 0

        self.llm = ChatOllama( model = self.llm_model_name )
        self.emb = OllamaEmbeddings( model = self.emb_model_name )

        # Setup RAG parameters
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        dimensions: int = len(self.emb.embed_query("dummy"))
        self.index = FAISS(embedding_function=self.emb, 
                           index=IndexFlatL2(dimensions), 
                           docstore=InMemoryByteStore(), 
                           index_to_docstore_id={}
                        )
            
        # Setup image generator
        self.client = OpenAI(base_url='https://external.api.recraft.ai/v1', api_key = env('RECRAFT_API_KEY'))

        # Setup database
        self.db = SQLDatabase.from_uri(
            f"postgresql+psycopg2://{env('DB_USER01')}:{env('DB_PASS01')}@localhost:{env('DB_PORT01')}/{env('DB_NAME01')}"
        )

        # Opening JSON file
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
        list_llm_models = []
        list_emb_models = []
        for ix, model in enumerate(models):
            model_name = model['name'].replace(':latest', '')
            if 'embed' not in model_name:
                list_llm_models.append(model_name)
            else:
                list_emb_models.append(model_name)

        return sorted( list_llm_models ), sorted( list_emb_models )

    def change_temperature(self, temp):
        """Transforms your PDF(s) into vector format and splits it(them) into chunks.
        Args:
            Float: A number between [0:1] for setting the temperature of the LLM
        Returns:
            Updated LLM
        """
        self.temperature = temp
        self.llm = ChatOllama(model=self.llm_model_name, temperature=self.temperature)

        # print(f'[Temp ] {self.temperature}')

        return

    def change_llm_model(self, model):
        """ Switches between Ollama LLMs
        Args:
            String: Name of the Ollama model to be used
        Returns:
            None
        """
        self.llm_model_name = model
        self.llm = ChatOllama(model=self.llm_model_name, temperature=self.temperature)

        print(f'[Model] {self.llm.model}')

        return

    def change_emb_model(self, model):
        """ Switches between Ollama Embedders
        Args:
            String: Name of the Ollama embedder model to be used
        Returns:
            None
        """
        self.emb_model_name = model
        self.emb = OllamaEmbeddings( model = self.emb_model_name )

        print(f'[Embed] {self.emb.model}')

        return

    def change_mode(self, chat_mode):
        """ Switch between interaction modes
        Args:
            String: Mode flag between 'LLM', 'SQL', 'RAG' and 'T2I'
        Returns:
            None
        """
        self.with_database = False
        self.with_context = False
        self.with_images = False
        if chat_mode == 'SQL':
            self.with_database = True
        elif chat_mode == 'RAG':
            self.with_context = True
        elif chat_mode == 'T2I':
            self.with_images = True

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
        elif db_name == 'Movies':
            DBUSER = env('DB_USER02')
            DBPASS = env('DB_PASS02')
            DBPORT = env('DB_PORT02')
            DBNAME = env('DB_NAME02')
            SUFFIX = 'postgresql+psycopg2'
        
        self.sql_examples = self.all_examples[db_name]
        db_uri = f"{SUFFIX}://{DBUSER}:{DBPASS}@localhost:{DBPORT}/{DBNAME}"

        self.db = SQLDatabase.from_uri(db_uri)
        print(f'[ DB  ] {db_name}')

        return

    def change_verbose(self, flag_verbose):
        self.verbose = True if flag_verbose else False

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
            response = re.search("(SELECT.*);", raw_query.replace("\n", " "))
            if response == None:
                response = raw_query
            else:
                response = f'{response.group(1)}'

            return response

        # Define the chain for generating the SQL query
        sql_chain = (
            RunnablePassthrough.assign(table_info=lambda _: db.get_table_info())
            | few_shot_prompt
            | self.llm
            | StrOutputParser()
        )
        raw_query = sql_chain.invoke({"input": query_txt, "top_k": 5})
        sql_query = extract_sql(raw_query)

        return sql_query

    def ingest_pdf(self, file):
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

        return None
    
    def respond(self, message, chat_history):
        # Mode: interaction with databases
        if self.with_database:
            answer_prompt = PromptTemplate.from_template( APPCFG.template_answer )
            sql_query = self.get_sql_chain(self.db, message)
            sql_run = self.db.run( sql_query )

            chain = (
                answer_prompt
                | self.llm
                | StrOutputParser()
            )

            output = f"[SQL] {sql_query};\n[Run] {sql_run};\n[LLM] " if self.verbose else ''
            for chunk in chain.stream(  {"question": message, "query": sql_query, "result": sql_run}  ):
                output = output + chunk

                yield output
        # Mode: text-to-image generation
        elif self.with_images:
            response    = self.client.images.generate(
                prompt          = message,
                style           = 'realistic_image',
                size            = '1024x1024',
                response_format = 'b64_json'
            )
            img_base64 = response.data[0].b64_json
            output = f'<img src="data:image/png;base64,{img_base64}">'

            yield output
        # Mode: interaction with PDF documents
        elif self.with_context:
            prompt = ChatPromptTemplate.from_template( APPCFG.template_context )
            
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
            for chunk in chain.stream( message ):
                output = output + chunk
                
            yield output
        # Mode: open chat with LLMs
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
    


