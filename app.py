import os
import json
import time
from flask import Flask, request, jsonify, Response
import google.generativeai as genai

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_MAPPING = {
    'gpt-4': 'gemini-1.5-pro',
    'gpt-4-turbo': 'gemini-1.5-pro',
    'gpt-3.5-turbo': 'gemini-1.5-flash',
    'gpt-4o': 'gemini-1.5-pro',
    'gpt-4o-mini': 'gemini-1.5-flash'
}

def convert_messages_to_gemini(messages):
    gemini_messages = []
    system_instruction = None
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            system_instruction = content
        elif role == 'user':
            gemini_messages.append({
                'role': 'user',
                'parts': [{'text': content}]
            })
        elif role == 'assistant':
            gemini_messages.append({
                'role': 'model',
                'parts': [{'text': content}]
            })
    
    return gemini_messages, system_instruction

def create_openai_response(gemini_response, model):
    try:
        content = gemini_response.text
    except:
        content = "Error generating response"
    
    return {
        'id': f'chatcmpl-{int(time.time())}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': content
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    }

def stream_openai_response(gemini_response, model):
    def generate():
        chunk_id = f'chatcmpl-{int(time.time())}'
        
        for chunk in gemini_response:
            try:
                text = chunk.text
                data = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': text},
                        'finish_reason': None
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
            except:
                continue
        
        final_data = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        messages = data.get('messages', [])
        model = data.get('model', 'gpt-3.5-turbo')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)
        stream = data.get('stream', False)
        
        gemini_model = MODEL_MAPPING.get(model, 'gemini-1.5-flash')
        gemini_messages, system_instruction = convert_messages_to_gemini(messages)
        
        generation_config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens,
        }
        
        model_instance = genai.GenerativeModel(
            model_name=gemini_model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        if stream:
            response = model_instance.generate_content(
                gemini_messages,
                stream=True
            )
            return stream_openai_response(response, model)
        else:
            response = model_instance.generate_content(gemini_messages)
            return jsonify(create_openai_response(response, model))
        
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error',
                'code': 500
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        'object': 'list',
        'data': [
            {'id': model, 'object': 'model', 'owned_by': 'google'}
            for model in MODEL_MAPPING.keys()
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'gemini_configured': GEMINI_API_KEY is not None})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'OpenAI-Compatible Gemini Proxy API',
        'status': 'running',
        'endpoints': {
            'chat': '/v1/chat/completions',
            'models': '/v1/models',
            'health': '/health'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
