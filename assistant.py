
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
        self.temperature = 0

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

    def change_temperature(self, temp):
        self.temperature = temp
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)

        print('Model temperature: ', self.temperature)

        return

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
                ("system", "You are a friendly assistant called Burrito. Answer with a casual tone to the given question! If you don't know the answer, just say that you don't know. "),
                ("human", question)
            ])
            chain = prompt | self.llm
            output = ""
            for chunk in chain.stream( {'text': prompt} ):
                output = output + chunk.content
                yield output
