from google import genai
import os 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(model="gemini-2.5-flash")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ("exit", "quit"):
        break

    response = chat.send_message(user_input)

    reply = response.text
    print("Bot:", reply)
