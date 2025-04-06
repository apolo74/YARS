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
    # Instantiate Assistant - it will attempt DB connection internally
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
                with gr.Column(scale=1, min_width=200):
                    st_void = gr.State()
                    
                    # Left panel: running configuration parameters
                    with gr.Row():
                        # Always include SQL mode in the choices
                        available_modes = ["LLM", "SQL", "RAG", "T2I"]
                        default_mode = "LLM"
                        # No conditional logic needed here anymore to build the list
                        # # Check if the assistant successfully connected to the DB
                        # if assistant.db_available: 
                        #     available_modes.insert(1, "SQL") # Insert SQL if DB is available
                        # # No need for fallback logic here, default is already LLM

                        chat_mode = gr.Radio(available_modes, value=default_mode, label="Chat mode")
                        # Add a notification if user clicks SQL when DB is unavailable
                        def handle_sql_selection(mode):
                            if mode == "SQL" and not assistant.db_available:
                                gr.Warning("Database connection failed or unavailable. SQL mode is disabled.")
                            # Call the original mode change logic
                            assistant.change_mode(mode)
                            # Return dummy value needed by Gradio for state changes potentially
                            return None 
                            
                        # chat_mode.change(assistant.change_mode, chat_mode, st_void)
                        chat_mode.change(handle_sql_selection, chat_mode, st_void) # Use the handler

                        @gr.render(inputs=chat_mode)
                        def show_split(chat_mode):
                            # LLM parameters:
                            dd_model = gr.Dropdown(choices=llm_models, value=assistant.llm_model_name, label="Model", interactive=True)
                            dd_model.change(assistant.change_llm_model, dd_model, st_void)
                            sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                            sl_temp.change(assistant.change_temperature, sl_temp, st_void)
                            # SQL parameters:
                            dd_mode = gr.Dropdown(choices=['Chinook', 'Movies'], value='Chinook', label='Database', interactive=True, visible=False)
                            dd_mode.change(assistant.change_database, dd_mode, st_void)
                            cb_verbose = gr.Checkbox(False, label='Verbose', visible=False )
                            cb_verbose.change(assistant.change_verbose, cb_verbose, st_void)
                            # RAG parameters:
                            dd_embedder = gr.Dropdown(choices=emb_models, value='mxbai-embed-large', label="Embedders", visible=False)
                            dd_embedder.change( assistant.change_emb_model, dd_embedder, st_void)
                            tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'], visible=False)
                            tb_file.upload(assistant.ingest_pdf, tb_file, st_void, show_progress='full')
                            tb_file.clear(assistant.clear_pdf)

                            # Determine visibility based on mode AND db availability for SQL
                            sql_params_visible = chat_mode == "SQL" and assistant.db_available
                            rag_params_visible = chat_mode == "RAG"
                            t2i_params_visible = chat_mode == "T2I"
                            llm_params_visible = chat_mode == "LLM"

                            # Apply visibility settings
                            dd_mode.visible = sql_params_visible
                            cb_verbose.visible = sql_params_visible

                            dd_embedder.visible = rag_params_visible
                            tb_file.visible = rag_params_visible

                            # Hide LLM params if T2I is selected
                            dd_model.visible = not t2i_params_visible 
                            sl_temp.visible = not t2i_params_visible

                # Main area: Chat interface
                with gr.Column(scale=5, elem_id='col'):
                    gr.ChatInterface(
                        fn=assistant.respond, 
                        type="messages", 
                        chatbot=gr.Chatbot(
                            height=500, 
                            type="messages",
                            # bubble_full_width=False,
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

    main_loop() # Call main_loop without arguments

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
