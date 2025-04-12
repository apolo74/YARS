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

    # Define available modes
    # Always include SQL mode in the choices, visibility handled later
    available_modes = ["LLM", "SQL", "RAG", "T2I"]
    default_mode = "LLM"

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
                    st_void = gr.State() # Keep state for potential future use
                    
                    # --- Define ALL parameter controls here --- 
                    # LLM parameters (Visible by default)
                    dd_model = gr.Dropdown(choices=llm_models, value=assistant.llm_model_name, label="Model", interactive=True, visible=True)
                    sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature", visible=True)
                    # SQL parameters (Hidden by default)
                    dd_mode = gr.Dropdown(choices=['Chinook', 'Movies'], value='Chinook', label='Database', interactive=True, visible=False)
                    cb_verbose = gr.Checkbox(False, label='Verbose', visible=False)
                    # RAG parameters (Hidden by default)
                    dd_embedder = gr.Dropdown(choices=emb_models, value='mxbai-embed-large', label="Embedders", interactive=True, visible=False)
                    tb_file = gr.File(label="File", file_count='single', file_types=['.pdf'], visible=False)
                    # T2I parameters (None currently defined)

                    # --- Add .change handlers --- 
                    dd_model.change(assistant.change_llm_model, dd_model, st_void)
                    sl_temp.change(assistant.change_temperature, sl_temp, st_void)
                    dd_mode.change(assistant.change_database, dd_mode, st_void)
                    cb_verbose.change(assistant.change_verbose, cb_verbose, st_void)
                    dd_embedder.change( assistant.change_emb_model, dd_embedder, st_void)
                    tb_file.upload(assistant.ingest_pdf, tb_file, st_void, show_progress='full')
                    tb_file.clear(assistant.clear_pdf)

                # Main area: Tabs and Chat interface
                with gr.Column(scale=5, elem_id='col'):
                    with gr.Tabs(elem_id="chat_tabs") as tabs:
                        for mode in available_modes:
                            gr.Tab(mode, id=mode)
                    
                    # Single Chat Interface below tabs
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
                        # additional_inputs=txt_sql # Keep commented out or remove if not needed
                    )

                    # --- Tab selection handler --- 
                    def handle_tab_select(evt: gr.SelectData): # Gets the selected tab info
                        selected_tab = evt.value # The value/label of the selected tab (e.g., "LLM", "SQL")
                        assistant.change_mode(selected_tab)

                        # Determine visibility based on selected tab AND db availability for SQL
                        llm_visible = selected_tab == "LLM"
                        sql_visible = selected_tab == "SQL" and assistant.db_available
                        rag_visible = selected_tab == "RAG"
                        t2i_visible = selected_tab == "T2I"
                        
                        # Issue warning for SQL if DB is unavailable
                        if selected_tab == "SQL" and not assistant.db_available:
                            gr.Warning("Database connection failed or unavailable. SQL mode features disabled.")
                        
                        # Return updates for ALL controls listed in tabs.select outputs
                        return {
                            dd_model: gr.update(visible=llm_visible or rag_visible), # Show model for LLM and RAG
                            sl_temp: gr.update(visible=llm_visible),             # Show temp only for LLM
                            dd_mode: gr.update(visible=sql_visible), 
                            cb_verbose: gr.update(visible=sql_visible),
                            dd_embedder: gr.update(visible=rag_visible),
                            tb_file: gr.update(visible=rag_visible),
                        }

                    # Connect the handler to the tabs' select event
                    # Outputs must list all components the handler function returns updates for
                    tabs.select(handle_tab_select, None, [dd_model, sl_temp, dd_mode, cb_verbose, dd_embedder, tb_file])
                    
                    # Set initial mode in assistant (needed because radio button default is gone)
                    assistant.change_mode(default_mode)

    return demo.launch()

if __name__ == "__main__":
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    main_loop() # Call main_loop without arguments

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
