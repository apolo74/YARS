from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def addition(a: float, b: float) -> float:
   """Add two numbers."""
   return a + b

@tool
def substraction(a: float, b: float) -> float:
   """Substract two numbers."""
   return a - b

@tool
def multiplication(a: float, b: float) -> float:
   """Multiply two numbers."""
   return a * b

@tool
def division(a: float, b: float) -> float:
   """Divide two numbers."""
   return a / b

def main_loop():
    llm = ChatOllama( model = 'llama3.2:3b' )

    tools = [addition, substraction, multiplication, division]
    llm_with_tools = llm.bind_tools(tools)
    # print( 'multiply: ', llm_with_tools.invoke("What is 3 * 12?").tool_calls )
    # print( 'add: ', llm_with_tools.invoke("What is 3 + 12?").tool_calls )
    # print(f'Multiply: {multiply.invoke({"a": 3, "b": 12})}')
    # print(f'Add: {add.invoke({"a": 3, "b": 12})}')

    # chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
    # print(chain.invoke("What's four times 23"))

    # query = "What is 3 * 12? Also, what is 11 + 49?"

    # messages = [HumanMessage(query)]

    # ai_msg = llm_with_tools.invoke(messages)
    # messages.append(ai_msg)

    # for tool_call in ai_msg.tool_calls:
    #     selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    #     tool_msg = selected_tool.invoke(tool_call)
    #     messages.append(tool_msg)

    # print('Final:\n', llm_with_tools.invoke(messages).content)

    try:
        while True:
            print(60 * '-', '\n')
            print( 'Enter your question (Ctrl+C to exit!) ' )
            query_txt = input( '[Question] ' )
            messages = [HumanMessage(query_txt)]
            # AI message
            ai_message = llm_with_tools.invoke(messages)
            # print(f'[AI] {ai_message}')
            messages.append(ai_message)
            # Tool message
            for tool_call in ai_message.tool_calls:
                selected_tool = {
                    "addition": addition, 
                    "substraction": substraction, 
                    "multiplication": multiplication,
                    "division": division
                }[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                print('[  Tool  ]', tool_msg.name.upper())
                messages.append(tool_msg)
            print(f'[ Answer ] ', end='', flush=True)
            print(llm_with_tools.invoke(messages).content)

            # for chunk in llm_with_tools.stream( query_txt ):
            #     print(chunk.content, end="", flush=True)
            print()
    except KeyboardInterrupt:
        print('Bye!')
    print()

    return

if __name__ == "__main__":
    print(80 * '-')
    print("Testing script".center(80))
    print(80 * '-')

    main_loop( )

    print(80 * '-')
    print("The end!".center(80))
    print(80 * '-')


