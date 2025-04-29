import re
import json
import torch
import logging
import os
import time
from collections import defaultdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_response, generate_questions_with_model, normalize_topic, logger, generate_single_question, find_questions_from_file, load_training_questions


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
api_logger = logging.getLogger("api")


app = Flask(__name__)
CORS(app)


conversation_history = {}  
recent_requests = {}  


has_cuda = torch.cuda.is_available()
if has_cuda:
    api_logger.info(f"API server will use GPU acceleration: {torch.cuda.get_device_name(0)}")
else:
    api_logger.warning("No GPU detected. API will run on CPU")

def get_client_id():
    
    ip = request.remote_addr
    agent = request.headers.get('User-Agent', '')
    return f"{ip}_{agent}"

def parse_topic_and_count(query_lower):
    
    pattern = re.compile(r"(generate|create|give|make|provide)\s+(?:(a|an|one|two|three|four|five|\d+)\s+)?question(s)?", re.IGNORECASE)
    match = pattern.search(query_lower)
    
    if match:
        qty_str = match.group(2)
        
        num_map = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        
        if qty_str is None:
            count = 3  
        else:
            qty_lower = qty_str.lower()
            if qty_lower in num_map:
                count = num_map[qty_lower]
            elif qty_str.isdigit():
                count = int(qty_str)
            else:
                count = 1  
                
        
        topic = pattern.sub("", query_lower)
        topic = re.sub(r"\b(about|on|regarding|for|related to|concerning|discussing|covering|of|in|from)\b", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"\s+", " ", topic).strip()
        
        return topic.strip(), count
    
    
    test_pattern = re.compile(r"(?:can you )?(?:generate|create|give|make|provide)?\s*(?:a\s+)?test\s+(?:of|with)\s+(\d+)\s+question(s)?", re.IGNORECASE)
    test_match = test_pattern.search(query_lower)
    
    if test_match:
        count = int(test_match.group(1))
        topic = test_pattern.sub("", query_lower)
        topic = re.sub(r"\b(about|on|regarding|for|related to|concerning|the topic of)\b", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"\s+", " ", topic).strip()
        return topic.strip(), count
    
    
    alt_pattern = re.compile(r"question(s)?\s+(?:about|on|regarding|for|related to)\s+(.+)", re.IGNORECASE)
    alt_match = alt_pattern.search(query_lower)
    
    if alt_match:
        topic = alt_match.group(2).strip()
        return topic, 3  
    
    
    if any(term in query_lower for term in ["question", "quiz", "test", "assessment"]):
        for term in ["question", "quiz", "test", "assessment"]:
            if term in query_lower:
                parts = query_lower.split(term)
                if len(parts) > 1:
                    topic = parts[-1].strip()
                    topic = re.sub(r"^\s*(about|on|regarding|for|related to|concerning|the topic of)\s+", "", topic, flags=re.IGNORECASE)
                    return topic.strip(), 3
    

    return query_lower.strip(), 0

def find_questions_from_file_unique(topic, count=10, threshold=0.6, used_questions=None):
    
    if used_questions is None:
        used_questions = set()
    
    questions = load_training_questions()
    if not questions:
        api_logger.warning("No training questions available")
        return []
        
    topic_normalized = normalize_topic(topic)
    api_logger.info(f"Looking for unique questions on topic: '{topic}' (normalized: '{topic_normalized}')")
    
    
    exact_matches = []
    partial_matches = []
    
    
    for item in questions:
        if 'topic' not in item or 'question' not in item:
            continue
            
        item_topic = normalize_topic(item.get('topic', '').lower())
        question_text = item['question'].lower()
        
        
        if question_text in used_questions:
            continue
            
        
        if item_topic == topic_normalized:
            exact_matches.append(item)
    
    
    if len(exact_matches) < count:
        topic_words = set(topic_normalized.split())
        
        for item in questions:
            if 'topic' not in item or 'question' not in item:
                continue
                
            
            if item in exact_matches or item['question'].lower() in used_questions:
                continue
                
            item_topic = normalize_topic(item['topic'])
            
            
            if topic_normalized in item_topic or item_topic in topic_normalized:
                partial_matches.append(item)
                continue
                
            
            item_topic_words = set(item_topic.split())
            if topic_words and item_topic_words and not topic_words.isdisjoint(item_topic_words):
                overlap = len(topic_words.intersection(item_topic_words))
                union = len(topic_words.union(item_topic_words))
                overlap_ratio = overlap / union
                if overlap_ratio >= threshold:
                    partial_matches.append(item)
    
    
    all_matches = exact_matches.copy()
    for item in partial_matches:
        if item not in all_matches:
            all_matches.append(item)
    
    api_logger.info(f"Found {len(exact_matches)} unused exact matches and {len(partial_matches)} unused partial matches")
    
    
    import random
    if all_matches:
        sample_size = min(count, len(all_matches))
        api_logger.info(f"Returning {sample_size} unique questions from training file")
        return random.sample(all_matches, sample_size)
    
    api_logger.warning(f"No unused matching questions found for topic: {topic}")
    return []

def cleanup_old_sessions():
    
    current_time = time.time()
    
    timeout = 3600  
    
    expired_clients = []
    for client_id, last_time in recent_requests.items():
        if current_time - last_time > timeout:
            expired_clients.append(client_id)
    
    if expired_clients:
        for client_id in expired_clients:
            if client_id in conversation_history:
                del conversation_history[client_id]
            del recent_requests[client_id]
        
        api_logger.info(f"Cleaned up {len(expired_clients)} expired sessions. Active sessions: {len(recent_requests)}")

@app.route('/chat', methods=['GET'])
def chat():
    try:
        user_query = request.args.get("query", "").strip()
        force_generation = request.args.get("force_generation", "false").lower() == "true"
        
        
        client_id = get_client_id()
        
        
        recent_requests[client_id] = time.time()
        
        
        if client_id not in conversation_history:
            conversation_history[client_id] = set()

        query_lower = user_query.lower()
        api_logger.info(f"Received query: {user_query} (Client: {client_id[:20]}...)")
        
        
        create_questions_pattern = re.compile(r'(?:create|generate|make|give|provide)\s+(\d+|one|two|three|four|five|several|some|a\s+few)\s+questions?\s+(?:about|on|for|related\s+to|concerning)\s+(.+)', re.IGNORECASE)
        create_match = create_questions_pattern.search(query_lower)
        
        if create_match:
            count_text = create_match.group(1).lower()
            topic = create_match.group(2).strip()
            topic = normalize_topic(topic)
            
        
            if not hasattr(app, 'topic_request_counts'):
                app.topic_request_counts = {}
                
            if topic in app.topic_request_counts:
                app.topic_request_counts[topic] += 1
                
                if app.topic_request_counts[topic] % 15 == 0:
                    logger.info(f"Resetting question history for topic: {topic}")
                    to_remove = []
                    for q in conversation_history[client_id]:
                        if topic.lower() in q.lower():
                            to_remove.append(q)
                    for q in to_remove:
                        conversation_history[client_id].remove(q)
            else:
                app.topic_request_counts[topic] = 1
            
            
            count_map = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
                "several": 3, "some": 3, "a few": 3
            }
            
            if count_text in count_map:
                count = count_map[count_text]
            else:
                try:
                    count = int(count_text)
                except ValueError:
                    count = 3  
            
            count = max(1, min(count, 10))  
            api_logger.info(f"Generating {count} questions on topic: {topic}")
            
            
            if count == 1:
                
                max_attempts = 5  
                for attempt in range(max_attempts):
                    question = generate_single_question(topic, force_generation=(force_generation or attempt > 0))
                    if not question[0].isupper() and question[0].isalpha():
                        question = question[0].upper() + question[1:]
                    if not question.endswith('?'):
                        question += '?'
                    
                    
                    question_lower = question.lower()
                    if question_lower not in conversation_history[client_id]:
                        conversation_history[client_id].add(question_lower)
                        api_logger.info(f"Generated unique question for client (attempt {attempt+1})")
                        break
                    elif attempt == max_attempts - 1:
                        
                        api_logger.warning(f"Could not find unique question after {max_attempts} attempts")
                
                return jsonify({"response": question}), 200
            else:
                
                
                
                unique_questions = []
                
                if not force_generation:
                
                    db_questions = find_questions_from_file_unique(
                        topic, 
                        count=count*2, 
                        used_questions=conversation_history[client_id]
                    )
                    
                    
                    for item in db_questions:
                        question_text = item["question"]
                        if not question_text.endswith('?'):
                            question_text += '?'
                        if not question_text[0].isupper() and question_text[0].isalpha():
                            question_text = question_text[0].upper() + question_text[1:]
                        
                        unique_questions.append(question_text)
                        if len(unique_questions) >= count:
                            break
                
                
                if len(unique_questions) < count:
                    
                    remaining = count - len(unique_questions)
                    api_logger.info(f"Need {remaining} more questions, generating with model")
                    
                    
                    generated = generate_questions_with_model(topic, count=remaining*2, force_generation=True)
                    
                    
                    for question in generated:
                        if question.lower() not in conversation_history[client_id]:
                            unique_questions.append(question)
                            if len(unique_questions) >= count:
                                break
                
                
                for question in unique_questions:
                    conversation_history[client_id].add(question.lower())
                
                api_logger.info(f"Found {len(unique_questions)} unique questions for topic '{topic}'")
                
                
                cleaned_questions = []
                for question in unique_questions:
                    if not question[0].isupper() and question[0].isalpha():
                        question = question[0].upper() + question[1:]
                    if not question.endswith('?'):
                        question += '?'
                    cleaned_questions.append(question)
                    
                formatted_questions = []
                for i, question in enumerate(cleaned_questions, 1):
                    formatted_questions.append(f"{i}. {question}")
                
                formatted_response = "\n\n".join(formatted_questions)
                
                return jsonify({"response": formatted_response}), 200
            
        
        elif any(x in query_lower for x in ["question", "quiz", "test", "assessment"]):
            topic, count = parse_topic_and_count(query_lower)
            api_logger.info(f"Extracted topic: '{topic}', count: {count}")
            
            
            topic = normalize_topic(topic)
            
            
            if len(topic.split()) <= 1 and len(topic) < 4:
                potential_topics = re.findall(r'\b[a-zA-Z]{4,}\b', user_query)
                if potential_topics:
                    potential_topics.sort(key=len, reverse=True)
                    topic = potential_topics[0]
                    topic = normalize_topic(topic)
                    api_logger.info(f"Using extracted topic: '{topic}' from query")
            
            
            count = min(max(count, 1), 10)
            
            
            if count == 1:
                
                max_attempts = 5
                for attempt in range(max_attempts):
                    question = generate_single_question(topic, force_generation=(force_generation or attempt > 0))
                    if not question[0].isupper() and question[0].isalpha():
                        question = question[0].upper() + question[1:]
                    if not question.endswith('?'):
                        question += '?'
                    
                    
                    if question.lower() not in conversation_history[client_id]:
                        conversation_history[client_id].add(question.lower())
                        break
                    elif attempt == max_attempts - 1:
                        api_logger.warning(f"Could not find unique question after {max_attempts} attempts")
                
                return jsonify({"response": question})
            else:
                
                unique_questions = []
                
                if not force_generation:
                    db_questions = find_questions_from_file_unique(
                        topic, 
                        count=count*2, 
                        used_questions=conversation_history[client_id]
                    )
                    
                    for item in db_questions:
                        question_text = item["question"]
                        if not question_text.endswith('?'):
                            question_text += '?'
                        if not question_text[0].isupper() and question_text[0].isalpha():
                            question_text = question_text[0].upper() + question_text[1:]
                        
                        unique_questions.append(question_text)
                        if len(unique_questions) >= count:
                            break
                
                if len(unique_questions) < count:
                    remaining = count - len(unique_questions)
                    generated = generate_questions_with_model(topic, count=remaining*2, force_generation=True)
                    
                    for question in generated:
                        if question.lower() not in conversation_history[client_id]:
                            unique_questions.append(question)
                            if len(unique_questions) >= count:
                                break
                
                
                for question in unique_questions:
                    conversation_history[client_id].add(question.lower())
                
                
                cleaned_questions = []
                for question in unique_questions:
                    if not question[0].isupper() and question[0].isalpha():
                        question = question[0].upper() + question[1:]
                    if not question.endswith('?'):
                        question += '?'
                    cleaned_questions.append(question)
                
                formatted_questions = []
                for i, question in enumerate(cleaned_questions, 1):
                    formatted_questions.append(f"{i}. {question}")
                
                formatted_response = "\n\n".join(formatted_questions)
                
                return jsonify({"response": formatted_response})
        else:
            
            response = get_response(user_query)
            return jsonify({"response": response})
            
    except Exception as e:
        api_logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "response": "I'm having trouble processing your request right now. Could you try rephrasing or asking something else?",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    api_logger.info("Starting API server")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)