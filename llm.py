import argparse
import os
import requests
import sys

# Global dictionary for model configurations

MODELS = {
    "flash-2.0-lite": {
        "REST_API_MODEL_NAME": "gemini-2.0-flash-lite",
        "REST_API_ENDPOINT_BASE": "https://generativelanguage.googleapis.com/v1beta/models",
        "API_KEY_ENV_VAR": "GEMINI_API_KEY"
    },
    "flash-2.0": {
        "REST_API_MODEL_NAME": "gemini-2.0-flash",
        "REST_API_ENDPOINT_BASE": "https://generativelanguage.googleapis.com/v1beta/models",
        "API_KEY_ENV_VAR": "GEMINI_API_KEY"
    },
    "flash-2.5": {
        "REST_API_MODEL_NAME": "gemini-2.5-flash-preview-05-20",
        "REST_API_ENDPOINT_BASE": "https://generativelanguage.googleapis.com/v1beta/models",
        "API_KEY_ENV_VAR": "GEMINI_API_KEY"
    },
    "flash-2.5-pro": {
        "REST_API_MODEL_NAME": "gemini-2.5-pro-preview-05-06",
        "REST_API_ENDPOINT_BASE": "https://generativelanguage.googleapis.com/v1beta/models",
        "API_KEY_ENV_VAR": "GEMINI_API_KEY"
    }
}

def send_prompt(model_key, prompt, context_files=None):
    if model_key not in MODELS:
        print(f"Model '{model_key}' not found in configuration.")
        sys.exit(1)

    model_conf = MODELS[model_key]
    api_key = os.getenv(model_conf["API_KEY_ENV_VAR"])
    
    if not api_key:
        print(f"API key not found in environment variable '{model_conf['API_KEY_ENV_VAR']}'.")
        sys.exit(1)

    endpoint = f"{model_conf['REST_API_ENDPOINT_BASE']}/{model_conf['REST_API_MODEL_NAME']}:generateContent"

    headers = {
        "Content-Type": "application/json"
    }

    contents = [{"role": "user", "parts": []}]

    # Add context from files
    if context_files:
        for file_path in context_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    contents[0]["parts"].append({"text": f"File: {os.path.basename(file_path)}\n\n{file_content}"})
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}")
                sys.exit(1)

    # Add user prompt
    contents[0]["parts"].append({"text": prompt})
    
    payload = {
        "contents": contents
    }

    params = {
        "key": api_key
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, params=params)
        response.raise_for_status()
        data = response.json()
        print("\nResponse:\n")
        print(data["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        print("Error communicating with the LLM API:", str(e))
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Talk to an LLM from the command line.")
    parser.add_argument("prompt", help="Text prompt to send to the language model.")
    parser.add_argument("--model", default="flash-2.0-lite", help="Model key to use: flash-2.0-lite flash-2.5-pro flash-2.5 flash-2.0 (default: flash-2.0-lite).")
    parser.add_argument("--files", nargs='*', help="List of text file paths to provide as context before the prompt.")

    args = parser.parse_args()
    send_prompt(args.model, args.prompt, args.files)

if __name__ == "__main__":
    main()
