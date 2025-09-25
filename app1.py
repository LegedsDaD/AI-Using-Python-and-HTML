from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama, LlamaCache
from flask_cors import CORS
import os
import sys

# --- Configuration (Best Practice: Use environment variables) ---
# Use a default model path but allow overriding via environment variable
MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "llama-2-7b-chat.Q4_K_M.gguf")
HOST_PORT = int(os.environ.get("HOST_PORT", 5000))
LLM_N_CTX = int(os.environ.get("LLM_N_CTX", 4096)) # Context window size
# Set to 999 to offload all layers to GPU (if available and compiled with CUDA/cuBLAS)
LLM_N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", 0)) 

# Create a Flask application instance.
app = Flask(__name__, template_folder='templates')
CORS(app) # Enable CORS for all routes

# Global variable for the model instance
llm: Llama | None = None

# --- Chatbot Model Setup Function ---
def load_llm_model():
    """Initializes and loads the Llama model, enabling the KV-Cache."""
    global llm
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.", file=sys.stderr)
        print("Please ensure the file is in the correct directory or adjust the MODEL_PATH environment variable.", file=sys.stderr)
        return None

    print(f"Loading local AI model: {MODEL_PATH} (n_ctx={LLM_N_CTX}, n_gpu_layers={LLM_N_GPU_LAYERS})...")
    try:
        # Load the model with configurable parameters
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=LLM_N_CTX,
            n_gpu_layers=LLM_N_GPU_LAYERS, # Use GPU if available and configured
            verbose=False # Keep model load output clean
        )
        
        # --- 🚀 SPEED IMPROVEMENT: ENABLE CACHE ---
        # The LlamaCache stores computations for the static part of the prompt 
        # (the system instruction and formatting), dramatically reducing processing time 
        # on all requests after the first one.
        LlamaCache(llm) 
        
        print("✅ Model loaded successfully! KV-Cache is ENABLED.")
        return llm
    except Exception as e:
        print(f"💥 Error loading model: {e}", file=sys.stderr)
        return None

# Load the model when the script starts
load_llm_model()

# --- Flask Routes ---

@app.route('/')
def serve_html():
    """Serves the main HTML template."""
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    """API endpoint to handle user messages and get chatbot responses."""
    
    if llm is None:
        # Use HTTP 503 Service Unavailable if the model failed to load
        return jsonify({"response": "The chatbot model is not available. Check server logs."}), 503

    # Use .get() with an empty dictionary to safely handle non-JSON or missing body
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Invalid JSON or missing request body.'}), 400
        
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
        
    # --- PROMPT TEMPLATE (ChatML format is often cleaner, but the current format works) ---
    prompt = (
        "A chat between a curious user and an AI assistant. The assistant gives helpful, concise, and polite answers to the user's questions. "
        "If the assistant does not know the answer, it says so.\n\n"
        f"### User:\n{user_message}\n\n"
        "### Assistant:\n"
    )
        
    try:
        # NOTE ON SPEED: While the KV-Cache helps with prompt processing, 
        # true speed perception comes from streaming. For non-streaming, 
        # keep this call as is. To enable streaming (better UX):
        # 
        # from flask import Response, stream_with_context
        # def generate():
        #     # ... (build prompt)
        #     for chunk in llm(prompt, ..., stream=True):
        #         yield chunk["choices"][0]["text"]
        # return Response(stream_with_context(generate()), mimetype='text/plain')

        output = llm(
            prompt,
            max_tokens=200,
            stop=["### User:", "\n###"], # Added "\n###" as a safer stop
            echo=False,
            temperature=0.7,
            top_k=40,
            top_p=0.95
        )
        
        # Extract the response, handling potential list access errors
        response_text = output.get("choices", [{}])[0].get("text", "").strip()
        
        # Log the conversation on the server side
        print(f"--- Conversation ---")
        print(f"👤 User: {user_message}") 
        print(f"🤖 Bot: {response_text}\n")
        # print(f"Tokens/sec: {output.get('usage', {}).get('tps', 'N/A')}") # Optional T/s logging

        return jsonify({'response': response_text})
        
    except Exception as e:
        print(f"💥 An error occurred during model inference: {e}", file=sys.stderr)
        # Return a generic 500 error for internal server issues
        return jsonify({"response": "An internal server error occurred while generating a response."}), 500

if __name__ == '__main__':
    # REMINDER: For production, use Gunicorn/uWSGI with a single worker (`-w 1`) 
    # for best performance and memory management.
    print(f"Starting Flask server on port {HOST_PORT}...")
    app.run(debug=True, port=HOST_PORT, host='0.0.0.0')
