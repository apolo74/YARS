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
            String: Mode flag between 'LLM', 'RAG' and 'T2I'
        Returns:
            None
        """
        self.with_context = False
        self.with_images = False
        
        if chat_mode == 'RAG':
            self.with_context = True
        elif chat_mode == 'T2I':
            self.with_images = True

        print(f'[Mode ] {chat_mode}')

        return

    def change_tone(self, tone_mode):
        self.tone_mode = tone_mode
        print(f'[Tone ] {self.tone_mode}')

        category_prompt = f"""Classify the following query into one of these categories:
            'technical', 'creative', or 'factual'.
        
            Query: {user_query}
        
            Return ONLY the category name and nothing else."""
        
        category_response = generate_text(category_prompt)
    
        category = category_response.lower()
        if "technical" in category:
            category = "technical"
        elif "creative" in category:
            category = "creative"
        else:
            category = "factual"
    
        print(f"Query classified as: {category}")
    
        if category == "technical":
            return handle_technical_query(user_query)
        elif category == "creative":
            return handle_creative_query(user_query)
        else:  
            return handle_factual_query(user_query)

    def change_verbose(self, flag_verbose):
        self.verbose = True if flag_verbose else False

        return

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
        # Mode: text-to-image generation
        if self.with_images:
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
    


