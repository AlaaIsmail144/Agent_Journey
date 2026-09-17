
# from google import genai
# import os
# client = genai.Client(api_key= os.environ.get("GENAI_API_KEY"))

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Explain what an AI agent is in one sentence."
# )

# print(response.text)


from google import genai
import os 
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

chat = client.chats.create(
    model="gemini-2.5-flash"
)

response = chat.send_message(
    message="Explain what an AI agent is in one sentence."
)

print(response.text)