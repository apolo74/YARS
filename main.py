
''' Chat with PDF documents using Ollama endpoints and LangChain
libraries. A GUI (Graphic User Interface) is provided and implemented
using the Gradio library.

Usage:    
        main.py [-h]

optional arguments:
    -h, --help                          Show this help message and exit
    
Author:   Boris Duran
Email:    boris@yodir.com
Created:  2024-11-09
'''

import requests
import gradio as gr

from assistant import Assistant

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
    PLACE_HOLDER = "Ask me anything!"
    with gr.Blocks(
        title='YARS',
        css=".contain { display: flex !important; flex-direction: column !important; }"
        "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
        "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
        "#col { height: calc(100vh - 170px) !important; "
        ".message-row img {margin: 0px !important;}"
        ".avatar-container img {padding: 0px !important;}"
    ) as all_blocks:                    
        gr.Markdown("""
                        # <div style="text-align: left; color:SteelBlue;"> [ YARS ] </div>
                        ## <div style="text-align: left"> Yet Another Retrieval Script</div>
                    """)
        with gr.Column(elem_classes=["container"]):   
            with gr.Row():
                # Left area: Parameters
                with gr.Column(scale=1, min_width=200):
                    st_void = gr.State()

                    # Top: Dropdown menu for choosing LLM models and temperature
                    with gr.Row():
                        dd_model = gr.Dropdown(models, value=assistant.model_name, label="Models", interactive=True)
                        dd_embedder = gr.Dropdown(["nomic", "phi3"], label="Embedders")
                        sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                    dd_model.change(assistant.change_model, dd_model, st_void)
                    sl_temp.change(assistant.change_temperature, sl_temp, st_void)  
                    
                    # Bottom: File ingestion                 
                    with gr.Row():
                        tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'])
                    tb_file.upload(assistant.ingest_pdf, tb_file, st_void, show_progress='full')
                    tb_file.clear(assistant.clear_pdf)

                # Main area: Chat interface
                with gr.Column(scale=5, elem_id='col'):
                    gr.ChatInterface(
                        fn=assistant.respond,
                        type="messages", 
                        chatbot=gr.Chatbot(
                            height=500, 
                            type="messages",
                            bubble_full_width=False, 
                            avatar_images=( ("images/human.png", "images/chatbot.png") ),
                            render=False, 
                            elem_id="chatbot"
                        ),
                        fill_height=True,
                        theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"),
                    )
            
    
    return all_blocks.launch()

if __name__ == '__main__':
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    main_loop()

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
