Project West Campus Chatbot (Prototype)
Only works locally so far with local text files for context (/data)
index.html just for visuals not actual final layout

How to Run This Project
1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install the required Python packages from requirements.txt
pip install -r requirements.txt

3. Create a file called .env in the folder and put your OpenAI API key inside it 
OPENAI_API_KEY=your-key-here

5. Run the ingest.py file to load the text files into the chatbot
python ingest.py

6. Start the backend server
uvicorn server:app --reload

7. Open the index.html file in a browser and chat with the bot

Stopping the Server
- Ctrl + C in the terminal

Future Plans:
Use Pinecone API for better vector database
Speed up response time