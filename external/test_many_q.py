import requests
import dotenv
import os
dotenv.load_dotenv()

api_key = os.getenv("PERSONAL_API_KEY")

message = {
    "messages": [
        {"role": "user", "content": "Who is the author of this song? https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    ],
}
for i in range(5):
    response = requests.post("http://localhost:8080/api/chat", json=message, headers={"Authorization": f"Bearer {api_key}"})
    
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print("Error:", response.status_code, response.text)