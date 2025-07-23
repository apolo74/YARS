import gradio as gr

# Define CSS to make the HTML component take up significant space
# This will apply to the container Gradio creates for the gr.HTML output
css = """
#trading-view-html-container {
    height: 90vh; /* 90% of viewport height */
    width: 100%;
    border: 2px solid orange; /* For visibility of the container */
}
/* Gradio's gr.HTML component wraps content in a div. Make that div take full height/width. */
#trading-view-html-container > div {
    height: 100%; 
    width: 100%;
}
"""

# HTML content for the TradingView widget
html_content = """
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
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
        \"container_id\": \"tradingview_widget_advanced_chart_v3\", /* Unique ID */
        \"support_host\": \"https://www.tradingview.com\"
    }
    </script>
</div>
"""

def main_app_v3():
    with gr.Blocks(css=css, title="TradingView Test V3") as demo:
        gr.Markdown("# Minimal TradingView Chart in Gradio (app_v3.py)")
        # Assign an elem_id to the HTML component for CSS targeting
        gr.HTML(html_content, elem_id="trading-view-html-container") 

    print("Launching app_v3.py...")
    demo.launch()

if __name__ == "__main__":
    main_app_v3()
