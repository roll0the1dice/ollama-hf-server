import requests
import json

OLLAMA_API_URL = "http://127.0.0.1:8434/api"

def generate_text(model_name, prompt_text, stream=False):
    """Generates text using the Ollama /api/generate endpoint."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model_name,
                "prompt": prompt_text,
                "stream": stream,
            },
            stream=stream # Important for requests library when stream=True
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        if stream:
            full_response = ""
            print("Streaming response:")
            # Process the stream line by line (ndjson)
            # b'{
            # "model":"phi3:mini",
            # "created_at":"2025-05-04T18:33:48.3695843Z",
            # "response":"Cur",
            # "done":false
            # }\n
            # {"model":"phi3:mini",
            # "created_at":"2025-05-04T18:33:48.3921177Z",
            # "response":"led","done":false}
            # \n
            # {"model":"phi3:mini",
            # "created_at":"2025-05-04T18:33:48.4156902Z",
            # "response":" upon","done":false
            # }\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.4415505Z","response":" the","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.4662332Z","response":" windows","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.4910883Z","response":"ill","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.5165911Z","response":",","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.5402036Z","response":" eyes","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.5650762Z","response":" half","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.5890142Z","response":"-","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.6132061Z","response":"closed","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.6369075Z","response":" in","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.6605086Z","response":" rep","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.6844025Z","response":"ose","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.7071736Z","response":",","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.7297942Z","response":"  ","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.752831Z","response":"\\n","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.7762357Z","response":"Sil","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.8000385Z","response":"hou","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.8242838Z","response":"ettes","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.8485161Z","response":" of","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.8723076Z","response":" fur","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.8965341Z","response":" against","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.920105Z","response":" glass","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.9448317Z","response":"y","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.9689437Z","response":" sk","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:48.9921555Z","response":"ies","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.0155572Z","response":" I","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.0391265Z","response":" disc","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.0630765Z","response":"ern","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.0863441Z","response":"ed","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.1104173Z","response":" to","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.1346243Z","response":" compose","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.1583777Z","response":";","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.1830329Z","response":"  ","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.2070504Z","response":"\\n","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.2314109Z","response":"Wh","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.2551823Z","response":"isk","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.2749676Z","response":"ers","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.3030075Z","response":" tw","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.3266776Z","response":"itch","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.3516278Z","response":"ing","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.375624Z","response":" soft","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.3991703Z","response":"ly","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.4221341Z","response":" as","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.4415628Z","response":" moon","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.4687653Z","response":"light","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.4923427Z","response":" we","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.5165802Z","response":"aves","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.5409752Z","response":" through","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.565143Z","response":" her","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.5891913Z","response":" coat","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.6131237Z","response":",","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.6365711Z","response":"   ","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.6599871Z","response":"\\n","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.6843286Z","response":"In","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.7085548Z","response":" dream","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.7321874Z","response":"s","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.7549626Z","response":" she","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.7792721Z","response":" pro","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.8027783Z","response":"w","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.8262598Z","response":"ls","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.8502238Z","response":" \xe2\x80\x94","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.8740709Z","response":" a","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.8978465Z","response":" vel","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.9169769Z","response":"vet","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.9452611Z","response":" p","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.9687628Z","response":"aw","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:49.9923395Z","response":" g","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:50.0158868Z","response":"ently","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:50.0395451Z","response":" af","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:50.0631975Z","response":"loat","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:50.0869141Z","response":".","done":false}\n{"model":"phi3:mini","created_at":"2025-05-04T18:33:50.1106585Z","response":"","done":true,"done_reason":"stop","context":[32010,29871,13,6113,263,3273,26576,1048,263,6635,29889,32007,29871,13,32001,29871,13,23902,839,2501,278,5417,453,29892,5076,4203,29899,15603,297,1634,852,29892,259,13,26729,10774,21158,310,3261,2750,12917,29891,2071,583,306,2313,824,287,304,27435,29936,259,13,8809,3873,414,3252,2335,292,4964,368,408,18786,4366,591,5989,1549,902,24296,29892,1678,13,797,12561,29879,1183,410,29893,3137,813,263,5343,5990,282,1450,330,2705,2511,3071,29889],"total_duration":1886713200,"load_duration":9442800,"prompt_eval_count":17,"prompt_eval_duration":133422900,"eval_count":74,"eval_duration":1742430100}\n'
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_chunk = json.loads(decoded_line)
                        response_part = json_chunk.get("response", "")
                        print(response_part, end="", flush=True)
                        full_response += response_part
                        # Check if generation is done
                        if json_chunk.get("done"):
                            print("\n--- Generation Complete ---")
                            # You can access final metadata here if needed
                            # print("Metadata:", json_chunk)
                    except json.JSONDecodeError:
                        print(f"\nError decoding JSON chunk: {decoded_line}")
            return full_response
        else:
            # Process non-streaming response
            result = response.json()
            print("Full response received:")
            print(result.get("response"))
            # print("Metadata:", result)
            return result.get("response")

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def chat(model_name, messages, stream=False):
    """Chats using the Ollama /api/chat endpoint."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/chat",
            json={
                "model": model_name,
                "messages": messages,
                "stream": stream,
            },
            stream=stream
        )
        response.raise_for_status()

        if stream:
            full_response_content = ""
            print("Streaming chat response:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_chunk = json.loads(decoded_line)
                        message_chunk = json_chunk.get("message", {})
                        content_part = message_chunk.get("content", "")
                        print(content_part, end="", flush=True)
                        full_response_content += content_part
                        if json_chunk.get("done"):
                             print("\n--- Chat Complete ---")
                             # print("Metadata:", json_chunk)

                    except json.JSONDecodeError:
                         print(f"\nError decoding JSON chunk: {decoded_line}")
            return {"role": "assistant", "content": full_response_content} # Return the assembled message
        else:
            result = response.json()
            print("Full chat response received:")
            assistant_message = result.get("message", {})
            print(assistant_message.get("content"))
            # print("Metadata:", result)
            return assistant_message

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    MODEL = "phi3:mini" # Make sure you have pulled this model

    # --- Generate Example (Non-Streaming) ---
    print("\n--- Generate Example (Non-Streaming) ---")
    prompt = "What is the capital of France?"
    generate_text(MODEL, prompt, stream=False)

    # --- Generate Example (Streaming) ---
    print("\n--- Generate Example (Streaming) ---")
    prompt = "Write a short poem about a cat."
    generate_text(MODEL, prompt, stream=True)

    # --- Chat Example (Streaming) ---
    print("\n--- Chat Example (Streaming) ---")
    chat_history = [
        {"role": "user", "content": "Hello! My name is Bob."}
    ]
    assistant_response = chat(MODEL, chat_history, stream=True)
    if assistant_response:
        chat_history.append(assistant_response) # Add assistant response for context

    print("\n--- Follow-up Chat (Streaming) ---")
    chat_history.append({"role": "user", "content": "What is my name?"})
    chat(MODEL, chat_history, stream=True) # Send the updated history