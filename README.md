Local AI Chatbot (Flask & Llama.cpp) 🤖
This project demonstrates how to build a simple, local chatbot application using a GGUF Large Language Model (LLM) powered by llama-cpp-python (which leverages the high-performance Llama.cpp library) and a Python Flask backend, all accessible via a clean HTML/CSS/JavaScript frontend.

The system is designed to run entirely on a local machine, offloading as many layers as possible to the GPU for faster inference if the environment is correctly configured.

✨ Features
Local Inference: Runs a Llama model directly on your CPU/GPU without needing an external API key.

Optimized Performance: Uses llama_cpp.LlamaCache to store computations for the static part of the prompt (system instructions), dramatically speeding up response times after the first request.

Modern Frontend: A responsive and accessible chat interface built with HTML, CSS, and vanilla JavaScript.

Asynchronous Communication: Handles sending messages, showing a "Bot is thinking..." loading indicator, and displaying responses via a fetch request to the backend API.

Configurable: Model path, context window size (n_ctx), and GPU layers (n_gpu_layers) can be configured via environment variables.

⚙️ Prerequisites
You must have Python 3.8+ installed.

1. Model File
You need a GGUF format model file. The application is configured by default to look for a file named:

llama-2-7b-chat.Q4_K_M.gguf

You can download a suitable model from Hugging Face or other model repositories and place it in the project's root directory.

2. Python Dependencies
The backend requires Flask, llama-cpp-python, and flask-cors.

Bash

pip install flask llama-cpp-python flask-cors
Note on llama-cpp-python: For best performance, especially with GPU acceleration, you may need to install this library with specific flags (e.g., for CUDA/cuBLAS). Consult the official llama-cpp-python documentation for installation instructions for your environment.

🚀 Setup & Running the Chatbot
1. Project Structure
Ensure your project directory is set up as follows.

/local-ai-chatbot
├── app1.py
├── index.html 
└── (your-model-file).gguf  # e.g., llama-2-7b-chat.Q4_K_M.gguf
IMPORTANT: The provided HTML file is named index 2.html, but the Flask application (app1.py) attempts to serve index.html from a templates folder. For this setup to work out-of-the-box, you should either:

Rename index 2.html to index.html and place it in a new templates directory.

Or, modify app1.py's serve_html function to render index 2.html and set the template_folder argument correctly in Flask(__name__, template_folder='.').

2. Configure (Optional)
You can use environment variables to customize the model loading and server port:

Variable	Default Value	Description
LLM_MODEL_PATH	llama-2-7b-chat.Q4_K_M.gguf	Path to your GGUF model file.
HOST_PORT	5000	The port the Flask server will listen on.
LLM_N_CTX	4096	The context window size for the model.
LLM_N_GPU_LAYERS	0	Number of layers to offload to the GPU (set to a high number like 999 to offload all if GPU is available).

Export to Sheets
Example command to run with GPU offloading (Linux/macOS):

Bash

export LLM_N_GPU_LAYERS=35
python app1.py
3. Start the Backend Server
Execute the Python script:

Bash

python app1.py
You should see output indicating the model is loading and the server is starting:

Loading local AI model: llama-2-7b-chat.Q4_K_M.gguf (n_ctx=4096, n_gpu_layers=0)...
✅ Model loaded successfully! KV-Cache is ENABLED.
Starting Flask server on port 5000...
 * Running on http://0.0.0.0:5000 (Press CTRL+C to quit)
4. Access the Chatbot
Open your web browser and navigate to:

http://127.0.0.1:5000/
The frontend will automatically attempt to connect to the backend's API endpoint at http://127.0.0.1:5000/chatbot. Type your message and hit Send to start the conversation.

🛠 Technology Stack
Backend: Python, Flask

LLM Integration: llama-cpp-python

Model Format: GGUF

Frontend: HTML5, CSS3 (Inter font), JavaScript (Vanilla)

Communication: REST (POST to /chatbot)
