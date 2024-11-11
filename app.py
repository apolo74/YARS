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
    models = assistant.get_models_list(local_models['models'])

    # Define UI
    PLACE_HOLDER = "Ask me something!"
    with gr.Blocks(
        title='YARS',
        css=".contain { display: flex !important; flex-direction: column !important; }"
        "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
        "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
        "#col { height: calc(100vh - 170px) !important; }"
        ".message-row img {margin: 0px !important;}"
        ".avatar-container img {padding: 0px !important;}"
    ) as demo:                        
        gr.Markdown("""
                        # <div style="text-align: left; color:DarkOrange;"> [ YARS ] </div>
                        ### <div style="text-align: left"> Yet Another Retrieval Script </div>
                    """)
        with gr.Column(elem_classes=["container"]):   
            with gr.Row():
                # Left area: Parameters
                with gr.Column(scale=1, min_width=200):
                    st_void = gr.State()
                    
                    # Top: Dropdown menu for choosing LLM models and temperature
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=models, value='llama3.2:1b', label="Model", interactive=True)
                        # dd_embedder = gr.Dropdown(["nomic", "phi3"], label="Embedders")
                        sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                    dd_model.change(assistant.change_model, dd_model, st_void)
                    sl_temp.change(assistant.change_temperature, sl_temp, st_void)

                    # Bottom: Radio buttons for choosing between chat with LLM or chat with database
                    with gr.Row():
                        chat_mode = gr.Radio(["LLM", "SQL"], value='LLM', show_label=False, info="Q&A with:") # label="Chat mode",
                        chat_mode.change(assistant.change_mode, chat_mode, st_void)
                        @gr.render(inputs=chat_mode)
                        def show_split(chat_mode):
                            dd_mode = gr.Dropdown(choices=['Chinook', 'Movies'], value='Chinook', label='Database', interactive=True, visible=False)
                            cb_verbose = gr.Checkbox(False, label='Verbose', visible=False )
                            if chat_mode == "SQL":
                                dd_mode.visible = True
                                cb_verbose.visible = True
                            # dd_mode.change(assistant.change_database, dd_mode, st_void)
                            cb_verbose.change(assistant.change_verbose, cb_verbose, st_void)

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


