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
        body, .gradio-container, .contain {
            background-color: #181818 !important;
            color: #f1f1f1 !important;
        }
        .contain { display: flex !important; flex-direction: column !important; }
        #component-0, #component-3, #component-10, #component-8  { height: 100% !important; }
        #chatbot { flex-grow: 1 !important; overflow: auto !important;}
        .chatbot.prose.md {opacity: 1.0 !important; color: #f1f1f1 !important;}
        #col { height: calc(100vh - 225px) !important; }
        .message-row img {margin: 0px !important;}
        .avatar-container img {padding: 0px !important;}
        
        /* Gradio input, output, and button styling for dark mode */
        .gradio-container .markdown-body, .gradio-container .prose {
            color: #f1f1f1 !important;
        }
    """
    html_content = """
    <div class=\"tradingview-widget-container\" style=\"height:100%; width:100%; display: flex; flex-direction: column;\">
        <div id=\"tradingview_widget_advanced_chart\" style=\"flex-grow: 1; width:100%;\"></div>
        <script type=\"text/javascript\" src=\"https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js\" async>
        {
            \"autosize\": true,
            \"symbol\": \"FX:EURUSD\",
            \"interval\": \"D\",
            \"timezone\": \"America/New_York\",
            \"theme\": \"dark\",
            \"style\": \"1\",
            \"locale\": \"en\",
            \"withdateranges\": true,
            \"hide_side_toolbar\": false,
            \"allow_symbol_change\": true,
            \"container_id\": \"tradingview_widget_advanced_chart\",
            \"support_host\": \"https://www.tradingview.com\"
        }
        </script>
    </div>
    """

    with gr.Blocks( title='YARS', css=css, fill_height=True ) as demo:                        
        gr.Markdown("""
            # <div> <span style="text-align: left; color:DarkOrange;"> [ YARS ] </span> <span style="color:#5F5F5F;"> Yet Another Retrieval Script </span></div>
        """)
        with gr.Tab("LLM"):
            with gr.Column(elem_classes=["container"]):   
                with gr.Row():
                    # Left area: Parameters
                    with gr.Column(scale=1, min_width=260):
                        st_void = gr.State()
                        
                        # Left panel: running configuration parameters
                        with gr.Row():
                            # LLM parameters:
                            dd_model = gr.Dropdown(choices=llm_models, value=assistant.llm_model_name, label="Model", interactive=True)
                            dd_model.change(assistant.change_llm_model, dd_model, st_void)
                            sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                            sl_temp.change(assistant.change_temperature, sl_temp, st_void)

                    # Main area: Chat interface
                    with gr.Column(scale=5, elem_id='col'):
                        gr.ChatInterface(
                            fn=assistant.respond, 
                            type="messages", 
                            chatbot=gr.Chatbot(
                                height=500, 
                                type="messages",
                                avatar_images=( ("images/human_dark.png", "images/chatbot_dark.png") ),
                                render=False,
                                elem_id="chatbot"
                            ),
                            # additional_inputs=txt_sql
                        )
        with gr.Tab("RAG"):
            with gr.Column(elem_classes=["container"]):   
                with gr.Row():
                    # Left area: Parameters
                    with gr.Column(scale=1, min_width=260):
                        st_void = gr.State()
                        
                        # Left panel: running configuration parameters
                        with gr.Row():
                            # LLM parameters:
                            dd_model = gr.Dropdown(choices=llm_models, value=assistant.llm_model_name, label="Model", interactive=True)
                            dd_model.change(assistant.change_llm_model, dd_model, st_void)
                            sl_temp = gr.Slider(value=0, minimum=0, maximum=1, step=0.1, label="Temperature")
                            sl_temp.change(assistant.change_temperature, sl_temp, st_void)
                        with gr.Row():
                            # RAG parameters:
                            dd_embedder = gr.Dropdown(choices=emb_models, value='mxbai-embed-large', label="Embedders")
                            dd_embedder.change( assistant.change_emb_model, dd_embedder, st_void)
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
                                avatar_images=( ("images/human_dark.png", "images/chatbot_dark.png") ),
                                render=False,
                                elem_id="chatbot"
                            ),
                            # additional_inputs=txt_sql
                        )
        with gr.Tab("T2I"):
            with gr.Column(elem_classes=["container"]):   
                with gr.Row():
                    # Left area: Parameters
                    with gr.Column(scale=1, min_width=260):
                        st_void = gr.State()
                        
                        # Left panel: running configuration parameters
                        with gr.Row():
                            gr.Image()
                    # Main area: Chat interface
                    with gr.Column(scale=5, elem_id='col'):
                        gr.ChatInterface(
                            fn=assistant.respond, 
                            type="messages", 
                            chatbot=gr.Chatbot(
                                height=500, 
                                type="messages",
                                avatar_images=( ("images/human_dark.png", "images/chatbot_dark.png") ),
                                render=False,
                                elem_id="chatbot"
                            ),
                        )
        with gr.Tab("Trading"):
            with gr.Column(elem_classes=["container"]):   
                with gr.Row():
                    # Left area: Parameters
                    with gr.Column(scale=1, min_width=260):
                        st_void = gr.State()
                        
                        # Left panel: running configuration parameters
                        with gr.Row():
                            gr.Markdown("Parameters")
                    # Main area: Trading interface
                    with gr.Column(scale=5, elem_id='col'):
                        gr.HTML( html_content )

    return demo.launch()

if __name__ == "__main__":
    print(80 * '-')
    print("YARS: Yet Another Retrieval Script".center(80))
    print(80 * '-')

    main_loop( )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')
