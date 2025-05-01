import gradio as gr
import requests

from utils.assistant import Assistant

def main_loop():
    """Main loop where all magic happens!
    Args:
      None
    Returns:
      None
    """
    assistant = Assistant()    
    local_models = requests.get('http://localhost:11434/api/tags').json()
    llm_models, emb_models = assistant.get_models_list(local_models['models'])

    # Define UI
    PLACE_HOLDER = "Ask me something!"
    css = """
        .contain { display: flex !important; flex-direction: column !important; }
        #component-0, #component-3, #component-10, #component-8  { height: 100% !important; }
        #chatbot { flex-grow: 1 !important; overflow: auto !important;}
        .chatbot.prose.md {opacity: 1.0 ! important}
        #col { height: calc(100vh - 170px) !important; }
        .message-row img {margin: 0px !important;}
        .avatar-container img {padding: 0px !important;}
    """
    with gr.Blocks( title='YARS', css=css ) as demo:                        
        gr.Markdown("""
            # <div style="text-align: left; color:DarkOrange;"> [ YARS ] </div>
            ### <div style="text-align: left"> Yet Another Retrieval Script </div>
        """)
        with gr.Column(elem_classes=["container"]):   
            with gr.Row():
                # Left area: Parameters
                with gr.Column(scale=1, min_width=260):
                    st_void = gr.State()
                    
                    # Left panel: running configuration parameters
                    with gr.Row():
                        chat_mode = gr.Radio(["LLM", "RAG", "T2I"], value='LLM', label="Chat mode")
                        chat_mode.change(assistant.change_mode, chat_mode, st_void)
                        @gr.render(inputs=chat_mode)
                        def show_split(chat_mode):
                            # LLM parameters:
                            dd_model = gr.Dropdown(choices=llm_models, value=assistant.llm_model_name, label="Model", interactive=True)
                            dd_model.change(assistant.change_llm_model, dd_model, st_void)
                            sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                            sl_temp.change(assistant.change_temperature, sl_temp, st_void)
                            with gr.Group():
                                tone_mode = gr.Radio(["Factual", "Technical", "Creative"], value='Factual', label="Tone", )
                                tone_mode.change(assistant.change_tone, tone_mode, st_void)
                            # RAG parameters:
                            dd_embedder = gr.Dropdown(choices=emb_models, value='mxbai-embed-large', label="Embedders", visible=False)
                            dd_embedder.change( assistant.change_emb_model, dd_embedder, st_void)
                            tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'], visible=False)
                            tb_file.upload(assistant.ingest_pdf, tb_file, st_void, show_progress='full')
                            tb_file.clear(assistant.clear_pdf)

                            if chat_mode == "RAG":
                                dd_embedder.visible = True
                                tb_file.visible     = True
                                tone_mode.visible   = False
                            elif chat_mode == "T2I":
                                dd_model.visible    = False
                                sl_temp.visible     = False
                                tone_mode.visible   = False
                            else:
                                dd_model.visible    = True
                                sl_temp.visible     = True

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
                        # additional_inputs=txt_sql
                    )
    return demo.launch()

if __name__ == "__main__":
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    main_loop( )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')


