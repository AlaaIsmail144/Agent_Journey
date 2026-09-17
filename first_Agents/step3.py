import os

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"File {path} not found"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read"},
                },
                "required": ["path"],
            },
        },
    },
]

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name=tool["function"]["name"],
        description=tool["function"]["description"],
        parameters=tool["function"]["parameters"],
    )
    for tool in TOOL_SCHEMAS
]

messages = [
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="What is inside notes.txt Summarize it in one line."
            )
        ],
    ),
]

while True:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=FUNCTION_DECLARATIONS)]
        ),
    )
    model_content = response.candidates[0].content
    messages.append(model_content)

    # No tool calls means the model is done and gave us a normal answer
    function_calls = response.function_calls
    if not function_calls:
        print(response.text)
        break

    function_responses = []
    for function_call in function_calls:
        args = function_call.args
        print(f"Model wants to run: read_file({args})")

        result = read_file(**args)

        function_responses.append(
            types.Part.from_function_response(
                name=function_call.name,
                response={"result": result},
            )
        )

    messages.append(types.Content(role="user", parts=function_responses))
    
    print(messages)
