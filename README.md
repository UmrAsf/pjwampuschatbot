Project West Campus Chatbot (Prototype)
Only works so far with local text files for context (/data)
index.html just for visuals not actual final layout
Now uses pinecone vector DB

How to Run This Project
1. Create and activate a virtual environment
- python3 -m venv .venv
- source .venv/bin/activate

2. Install the required Python packages from requirements.txt
- pip install -r requirements.txt

3. Create a file called .env and put OpenAI API key inside (look at .env.example as an example): 
- OPENAI_API_KEY=your-key-here
- PINECONE_API_KEY=your-pinecone-key

4. Create Pinecone Index called "projectwampus"
- use text-embedding-small configuration with dimension 1536    

5. Run the ingest.py file to load the text files into the chatbot
- python ingest.py

6. Start the backend server in Terminal:
- uvicorn server:app --reload

7. Open the index.html file in a browser and chat with the bot

Stopping the Server
- Ctrl + C in the terminal
