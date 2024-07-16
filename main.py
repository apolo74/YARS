
''' Chat with PDF documents using Ollama endpoints and LangChain
libraries. A GUI (Graphic User Interface) is provided and implemented
using the Gradio library.

Usage:    
        main.py [-h] -d DOCUMENT

optional arguments:
    -h, --help                          Show this help message and exit
    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-07-09
'''

import os
import requests
import gradio as gr

from assistant import Assistant

css = """
.container {
    height: 15vh;
}
#chatbot {
    flex-grow: 1 !important;
    overflow: auto !important;
}
"""
#col { height: calc(100vh - 112px - 16px) !important; }
'''
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

    return models_list

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
'''
def main_loop():
    """Main loop where all magic happens!
    Args:
      None
    Returns:
      None
    """
    assistant = Assistant()    
    local_models = requests.get('http://localhost:11434/api/tags').json()
    models = assistant.get_models_list(local_models['models'])

    # Define UI
    PLACE_HOLDER = "Ask me something about your docs!"
    with gr.Blocks(css=css) as all_blocks:
        with gr.Column(elem_classes=["container"]):   
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    with gr.Row():
                        gr.Markdown("""
                                        # <div style="text-align: right"> YARS </div>
                                        <div style="text-align: right"> Yet Another RAG Script</div>
                                    """)
                    with gr.Row():
                        tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'])
                        # tb_url = gr.Textbox(label="URL")
                    with gr.Row():
                        dd_model = gr.Dropdown(models, value=assistant.model_name, label="Models", interactive=True)
                        dd_embedder = gr.Dropdown(["nomic", "phi3"], label="Embedders")
                        sl_temperature = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                with gr.Column(scale=5):
                    gr.ChatInterface(
                        fn=assistant.respond,
                        chatbot=gr.Chatbot(elem_id="chatbot", render=False, bubble_full_width=False),
                        clear_btn=None,
                        retry_btn=None,
                        undo_btn=None,
                        submit_btn="Submit",
                        fill_height=True,
                        theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"),
                    )
            sl_temperature.change(assistant.change_temperature, sl_temperature)
            dd_model.change(assistant.change_model, dd_model)
            tb_file.upload(assistant.ingest_pdf, tb_file, show_progress='full')
            tb_file.clear(assistant.clear_pdf)
    
    return all_blocks.launch()

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another RAG Script".center(80))
    print(80 * '-')

    main_loop()

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
