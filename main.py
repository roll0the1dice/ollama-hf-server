# main.py
import os
import time
import json
import torch
import uvicorn
import threading
import asyncio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import gc # Garbage collector for clearing cache

# --- Configuration ---
# Define models you want to serve here
# Key: Name used in API (e.g., "llama3:latest")
# Value: Hugging Face model identifier or local path
AVAILABLE_MODELS = {
    # Replace with models you have access to and resources for
    # Add more models here
    'distilgpt2': 'distilgpt2',
    'distilgpt2:latest': 'distilgpt2'  # Add :latest variant
}

# Global storage for loaded models and tokenizers
loaded_models: Dict[str, Dict[str, Any]] = {}

# Global variable for distilgpt2 model
DISTILGPT2_MODEL: Dict[str, Any] = None

# --- Pydantic Models (Matching Ollama Spec) ---

class ModelDetails(BaseModel):
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None

class ModelInfo(BaseModel):
    name: str  # Keep name field for OpenWebUI compatibility
    model: str  # Keep model field for Ollama compatibility
    modified_at: str
    size: int
    digest: str
    details: ModelDetails

class ListModelsResponse(BaseModel):
    models: List[ModelInfo]

class ShowModelRequest(BaseModel):
    name: str

class ShowModelResponse(BaseModel):
    license: Optional[str] = None
    modelfile: Optional[str] = None
    parameters: Optional[str] = None
    template: Optional[str] = None
    details: ModelDetails

class GenerationRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = {}
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    format: Optional[str] = None
    raw: bool = False

class GenerationResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: bool = False
    options: Optional[Dict[str, Any]] = {}
    format: Optional[str] = None

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Dict[str, str]
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done_reason: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


# --- Model Loading Logic ---

def get_model_config(hf_model_config):
    """Extract basic parameters from HuggingFace model config."""
    params = {}
    if hasattr(hf_model_config, "num_hidden_layers"):
        params["num_layer"] = hf_model_config.num_hidden_layers
    if hasattr(hf_model_config, "hidden_size"):
        params["hidden_size"] = hf_model_config.hidden_size
    if hasattr(hf_model_config, "num_attention_heads"):
        params["num_attention_heads"] = hf_model_config.num_attention_heads
    if hasattr(hf_model_config, "vocab_size"):
        params["vocab_size"] = hf_model_config.vocab_size
    # Add more relevant parameters as needed

    param_str = "\n".join([f"{k} {v}" for k, v in params.items()])
    return param_str

def load_model(model_name: str):
    # Handle :latest suffix
    base_name = model_name.split(':')[0] if ':' in model_name else model_name
    
    # if model_name not in AVAILABLE_MODELS:
    #     raise HTTPException(status_code=404, detail=f"Model '{model_name}' not available.")
    # if model_name in loaded_models:
    #     return loaded_models[model_name]

    hf_identifier = 'distilgpt2' # AVAILABLE_MODELS[base_name]
    print(f"Loading model: {base_name} ({hf_identifier})...")
    start_time = time.time()

    try:
        # Configure device (prefer CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32 # Use float16 on GPU

        tokenizer = AutoTokenizer.from_pretrained(hf_identifier)
        model = AutoModelForCausalLM.from_pretrained(
            hf_identifier,
            torch_dtype=torch_dtype,
            device_map="auto",  # Let accelerate handle device placement if multiple GPUs
            # low_cpu_mem_usage=True, # May help on systems with low RAM
            trust_remote_code=True # Be cautious with this
            # Add quantization config here if needed (e.g., load_in_4bit=True requires bitsandbytes)
        )
        model.eval() # Set to evaluation mode

        # --- Get Model Details (Best Effort) ---
        try:
            # Attempt to estimate size (very rough)
            mem_bytes = model.get_memory_footprint()
            size_estimate = mem_bytes
        except Exception:
            size_estimate = 0 # Placeholder

        hf_config = model.config
        parameters_str = get_model_config(hf_config)
        quant_level = "FP16" if torch_dtype == torch.float16 else "FP32" # Basic guess
        if hasattr(model, "quantization_method"): # For bitsandbytes etc.
             quant_level = str(model.quantization_method)

        details = ModelDetails(
            parameter_size=f"{hf_config.num_parameters / 1e9:.2f}B" if hasattr(hf_config, 'num_parameters') else "Unknown",
            quantization_level=quant_level
        )
        template_str = tokenizer.chat_template if hasattr(tokenizer, "chat_template") else "No chat template found"

        loaded_models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device, # Store primary device
            "hf_identifier": hf_identifier,
            "load_time": time.time() - start_time,
            "size_bytes": size_estimate, # Store estimated size
            "details": details,
            "parameters": parameters_str,
            "template": template_str,
            "license": getattr(hf_config, "license", None),
        }
        print(f"Model {model_name} loaded in {loaded_models[model_name]['load_time']:.2f}s")
        print(f"Estimated Size: {size_estimate / (1024**3):.2f} GB")
        # Clear GPU cache after loading if using CUDA
        if device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        return loaded_models[model_name]

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {e}")


# --- FastAPI Application ---
app = FastAPI(title="Hugging Face Ollama Compatible API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allows all hosts
)

@app.middleware("http")
async def redirect_ollama_port(request: Request, call_next):
    """Redirect requests from Ollama's default port to our port."""
    # Remove port redirection since we're only running on 11434
    return await call_next(request)

@app.on_event("startup")
async def startup_event():
    # Load distilgpt2 model on startup
    global DISTILGPT2_MODEL
    try:
        DISTILGPT2_MODEL = load_model('distilgpt2')
        print("Successfully loaded distilgpt2 model")
    except Exception as e:
        print(f"Failed to load distilgpt2 model: {e}")
        raise e

    # Pre-load models on startup (optional, can load on first request too)
    for name in AVAILABLE_MODELS.keys():
        try:
            load_model(name)
            print(f"Successfully pre-loaded: {name}")
        except Exception as e:
            print(f"Failed to pre-load model {name}: {e}")


# --- API Endpoints ---

@app.get("/api/tags", response_model=ListModelsResponse)
async def list_models_detailed():
    """Lists models available through this server."""
    models_info = []
    for name, data in loaded_models.items():
        now = datetime.utcnow().isoformat() + "Z"
        models_info.append(
            ModelInfo(
                name=name,  # Include name field for OpenWebUI
                model=name,  # Include model field for Ollama
                modified_at=now,
                size=data.get("size_bytes", 0),
                digest=data.get("hf_identifier", "unknown")[:12],
                details=data.get("details", ModelDetails())
            )
        )
    # Also list models defined but not yet loaded
    for name in AVAILABLE_MODELS:
        if name not in loaded_models:
             models_info.append(
                ModelInfo(
                    name=name,  # Include name field for OpenWebUI
                    model=name,  # Include model field for Ollama
                    modified_at=datetime.utcnow().isoformat() + "Z",
                    size=0,
                    digest="not_loaded",
                    details=ModelDetails()
                )
             )
    return ListModelsResponse(models=models_info)

@app.post("/api/show", response_model=ShowModelResponse)
async def show_model(request: ShowModelRequest):
    """Show details for a specific model."""
    if request.name not in AVAILABLE_MODELS:
         raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found in configuration.")

    if request.name not in loaded_models:
        try:
            print(f"/api/show triggering load for: {request.name}")
            load_model(request.name)
        except HTTPException as e:
            raise e
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to load model {request.name} on demand: {e}")

    model_data = loaded_models[request.name]
    return ShowModelResponse(
        license=model_data.get("license"),
        parameters=model_data.get("parameters"),
        template=model_data.get("template"),
        details=model_data.get("details", ModelDetails())
    )


# --- Stop Sequence Handling ---
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return torch.BoolTensor([True], device=input_ids.device)
        return torch.BoolTensor([False], device=input_ids.device)

# --- Generation Logic ---

async def _generate_stream(
    model, tokenizer, inputs, streamer, generation_kwargs, model_name, start_time_ns
):
    """Helper function to run model.generate in a thread for streaming."""
    try:
        model.generate(**inputs, streamer=streamer, **generation_kwargs)
    except Exception as e:
        print(f"Error during generation thread: {e}")
        # You might want to signal the error back to the main stream,
        # potentially by yielding a specific error message or closing the streamer.

async def generate_stream_response(
    request: Union[GenerationRequest, ChatRequest],
    model_data: Dict[str, Any],
    inputs: Dict[str, torch.Tensor],
    generation_kwargs: Dict[str, Any],
):
    """Handles streaming responses for /generate and /chat"""
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_name = request.model
    device = model.device

    if isinstance(request, ChatRequest):
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (request.messages or [])])
        input_text += "\nassistant:"
    else:
        input_text = request.prompt
        if request.system:
            input_text = f"System: {request.system}\nUser: {request.prompt}"

    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(**model_inputs, streamer=streamer, **generation_kwargs),
    )
    generation_thread.start()

    generated_text = ""
    output_token_ids = []
    eval_count = 0
    start_time_ns = time.time_ns()
    load_duration_ns = int(model_data.get("load_time", 0) * 1e9)

    try:
        for new_text in streamer:
            if new_text:
                created_at = datetime.utcnow().isoformat() + "Z"
                generated_text += new_text
                eval_count += 1

                new_token_ids = tokenizer.encode(new_text, add_special_tokens=False)
                output_token_ids.extend(new_token_ids)

                if isinstance(request, ChatRequest):
                    response = ChatResponse(
                        model=model_name,
                        created_at=created_at,
                        message={"role": "assistant", "content": new_text},
                        done=False
                    )
                else:
                    response = GenerationResponse(
                        model=model_name,
                        created_at=created_at,
                        response=new_text,
                        done=False
                    )
                
                yield f"{response.model_dump_json()}\r\n"

    except Exception as e:
        print(f"Error receiving from streamer: {e}")
        error_response = {
            "error": f"Generation failed: {e}",
            "done": True
        }
        yield f"{json.dumps(error_response)}\n\n"
        return

    finally:
        generation_thread.join(timeout=5)
        if generation_thread.is_alive():
            print("Warning: Generation thread did not exit cleanly.")

        total_duration_ns = time.time_ns() - start_time_ns
        created_at = datetime.utcnow().isoformat() + "Z"
        prompt_eval_count = model_inputs["input_ids"].shape[-1]

        if isinstance(request, ChatRequest):
            final_response = ChatResponse(
                model=model_name,
                created_at=created_at,
                message={"role": "assistant", "content": ""},
                done=True,
                total_duration=total_duration_ns,
                load_duration=load_duration_ns,
                prompt_eval_count=prompt_eval_count,
                eval_count=eval_count,
                done_reason="stop"
            )
        else:
            final_response = GenerationResponse(
                model=model_name,
                created_at=created_at,
                response="",
                done=True,
                context=output_token_ids,
                total_duration=total_duration_ns,
                load_duration=load_duration_ns,
                prompt_eval_count=prompt_eval_count,
                eval_count=eval_count
            )
        
        yield f"{final_response.model_dump_json()}\r\n"

async def generate_non_stream_response(
     request: Union[GenerationRequest, ChatRequest],
     model_data: Dict[str, Any],
     inputs: Dict[str, torch.Tensor],
     generation_kwargs: Dict[str, Any],
):
    """Handles non-streaming responses for /generate and /chat"""
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    model_name = request.model
    device = model.device

    start_time_ns = time.time_ns()
    load_duration_ns = int(model_data.get("load_time", 0) * 1e9)

    if isinstance(request, ChatRequest):
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.messages])
        input_text += "\nassistant:"
    else:
        input_text = request.prompt
        if request.system:
            input_text = f"System: {request.system}\nUser: {request.prompt}"

    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    total_duration_ns = time.time_ns() - start_time_ns

    input_length = model_inputs["input_ids"].shape[1]
    output_token_ids = outputs[0][input_length:].tolist()
    response_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
    created_at = datetime.utcnow().isoformat() + "Z"

    prompt_eval_count = input_length
    eval_count = len(output_token_ids)
    eval_duration = total_duration_ns // 1_000_000  # Convert to milliseconds
    prompt_eval_duration = load_duration_ns // 1_000_000  # Convert to milliseconds

    if isinstance(request, ChatRequest):
        return ChatResponse(
            model=model_name,
            created_at=created_at,
            message={"role": "assistant", "content": response_text},
            done=True,
            total_duration=total_duration_ns,
            load_duration=load_duration_ns,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=eval_count,
            eval_duration=eval_duration,
            done_reason="stop"
        )
    else:
        return GenerationResponse(
            model=model_name,
            created_at=created_at,
            response=response_text,
            done=True,
            context=output_token_ids,
            total_duration=total_duration_ns,
            load_duration=load_duration_ns,
            prompt_eval_count=prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=eval_count,
            eval_duration=eval_duration
        )


def prepare_generation_kwargs(request: Union[GenerationRequest, ChatRequest], tokenizer):
    """Prepare kwargs for model.generate from request options."""
    options = request.options or {}
    kwargs = {}

    # Map Ollama options to HF generate args
    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]
        if options["temperature"] > 0:
             kwargs["do_sample"] = True # Sampling must be enabled for temperature
        else:
             kwargs["do_sample"] = False # Temperature 0 often means greedy
    else:
        kwargs["do_sample"] = True # Default to sampling if not specified

    if "top_k" in options:
        kwargs["top_k"] = options["top_k"]
        kwargs["do_sample"] = True # Ensure sampling is on for top_k
    if "top_p" in options:
        kwargs["top_p"] = options["top_p"]
        kwargs["do_sample"] = True # Ensure sampling is on for top_p
    if "num_predict" in options and options["num_predict"] > 0:
        # Be careful with max_length vs max_new_tokens
        # Let's prefer max_new_tokens if num_predict is given
        kwargs["max_new_tokens"] = options["num_predict"]
    else:
        kwargs["max_new_tokens"] = 256 # Default max tokens

    # Stopping criteria
    stop_sequences = options.get("stop", [])
    if stop_sequences:
        stop_token_ids = []
        for seq in stop_sequences:
            # Encode stop sequences. Be mindful of tokenization differences.
            token_ids = tokenizer.encode(seq, add_special_tokens=False)
            if token_ids:
                stop_token_ids.append(token_ids[-1]) # Often stop on the last token of sequence

        if stop_token_ids:
             kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

    # Add other relevant parameters (e.g., repetition_penalty)
    if "repetition_penalty" in options:
        kwargs["repetition_penalty"] = options["repetition_penalty"]

    # Set EOS token ID for generation termination
    kwargs["eos_token_id"] = tokenizer.eos_token_id

    return kwargs

@app.post("/api/generate")
async def generate_completion(request: GenerationRequest, http_request: Request):
    """Generate text based on a prompt (compatible with Ollama generate)."""
    if request.prompt is None:
         raise HTTPException(status_code=400, detail="Prompt is required for /api/generate")

    model_data = load_model(request.model)
    tokenizer = model_data["tokenizer"]
    device = model_data["device"]

    full_prompt = request.prompt
    if request.system:
         full_prompt = f"System: {request.system}\nUser: {request.prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    generation_kwargs = prepare_generation_kwargs(request, tokenizer)

    if request.stream:
        return StreamingResponse(
            generate_stream_response(request, model_data, inputs, generation_kwargs),
            media_type="application/x-ndjson"
        )
    else:
        response_data = await generate_non_stream_response(request, model_data, inputs, generation_kwargs)
        return response_data

@app.post("/api/chat")
async def generate_chat_completion(request: ChatRequest, http_request: Request):
    """Generate chat completion based on message history (compatible with Ollama chat)."""
    if not request.messages:
         raise HTTPException(status_code=400, detail="Messages are required for /api/chat")

    # Use the globally loaded distilgpt2 model
    global DISTILGPT2_MODEL
    if DISTILGPT2_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded") # TODO: Remove this
        
    model = DISTILGPT2_MODEL["model"]
    tokenizer = DISTILGPT2_MODEL["tokenizer"]
    model_name = 'distilgpt2'
    device = model.device

    if isinstance(request, ChatRequest):
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in (request.messages or [])])
        input_text += "\nassistant:"
    else:
        input_text = request.prompt
        if request.system:
            input_text = f"System: {request.system}\nUser: {request.prompt}"

    model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
    generation_kwargs = prepare_generation_kwargs(request, tokenizer)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=dict(**model_inputs, streamer=streamer, **generation_kwargs),
    )
    generation_thread.start()

    start_time_ns = time.time_ns()
    load_duration_ns = time.time_ns()

    if request.stream:
        async def generate_stream():
            generated_text = ""
            output_token_ids = []
            eval_count = 0
            
            try:
                # Create a queue to handle the stream
                queue = asyncio.Queue()
                loop = asyncio.get_event_loop()
                
                def put_in_queue():
                    for text in streamer:
                        future = asyncio.run_coroutine_threadsafe(queue.put(text), loop)
                        future.result()  # Wait for the put to complete
                    future = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                    future.result()  # Wait for the final put to complete
                
                # Start the generation in a separate thread
                threading.Thread(target=put_in_queue, daemon=True).start()
                
                # Process the stream
                while True:
                    new_text = await queue.get()
                    if new_text is None:
                        break
                        
                    if new_text:
                        created_at = datetime.utcnow().isoformat() + "Z"
                        generated_text += new_text
                        eval_count += 1

                        new_token_ids = tokenizer.encode(new_text, add_special_tokens=False)
                        output_token_ids.extend(new_token_ids)

                        response = {
                            "model": model_name,
                            "created_at": created_at,
                            "message": {
                                "role": "assistant",
                                "content": new_text
                            },
                            "done": False
                        }
                        
                        yield f"{json.dumps(response)}\r\n"

                # Send final response with statistics
                total_duration_ns = time.time_ns() - start_time_ns
                prompt_eval_count = model_inputs["input_ids"].shape[-1]
                
                final_response = {
                    "model": model_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": ""
                    },
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": total_duration_ns,
                    "load_duration": 8305200,
                    "prompt_eval_count": prompt_eval_count,
                    "prompt_eval_duration": 344353100,
                    "eval_count": eval_count,
                    "eval_duration": total_duration_ns - 8305200
                }
                
                yield f"{json.dumps(final_response)}\r\n"

            except Exception as e:
                print(f"Error during generation: {e}")
                error_response = {
                    "error": f"Generation failed: {e}",
                    "done": True
                }
                yield f"{json.dumps(error_response)}\r\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming response
        try:
            with torch.no_grad():
                outputs = model.generate(**model_inputs, **generation_kwargs)

            total_duration_ns = time.time_ns() - start_time_ns
            input_length = model_inputs["input_ids"].shape[1]
            output_token_ids = outputs[0][input_length:].tolist()
            response_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)
            
            return {
                "model": model_name,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": total_duration_ns,
                "load_duration": 0,
                "prompt_eval_count": input_length,
                "prompt_eval_duration": 0,  # TODO: Calculate actual duration
                "eval_count": len(output_token_ids),
                "eval_duration": total_duration_ns - 0
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

@app.get("/")
async def read_root():
    return {"message": "Hugging Face Ollama Compatible API is running"}

@app.get("/api/version")
async def get_version():
    """Return version information."""
    return {
        "version": "0.1.0",
        "api_version": "v1"
    }

@app.get("/ollama/api/version")
async def ollama_version():
    """Proxy endpoint for Ollama's version API."""
    return {
        "version": "0.1.0",
        "api_version": "v1"
    }

@app.get("/ollama/api/tags")
async def ollama_tags():
    """Proxy endpoint for Ollama's tags API."""
    return await list_models_detailed()

@app.post("/ollama/api/generate")
async def ollama_generate(request: GenerationRequest):
    """Proxy endpoint for Ollama's generate API."""
    return await generate_completion(request, None)

@app.post("/ollama/api/chat")
async def ollama_chat(request: ChatRequest):
    """Proxy endpoint for Ollama's chat API."""
    return await generate_chat_completion(request, None)

@app.post("/ollama/api/show")
async def ollama_show(request: ShowModelRequest):
    """Proxy endpoint for Ollama's show API."""
    return await show_model(request)

# --- Main Execution ---
if __name__ == "__main__":
    # Set environment variables for tokenizers if needed
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run the server on Ollama's default port only
    uvicorn.run(app, host="0.0.0.0", port=11434)