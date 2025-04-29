import faiss
import json
import sqlite3
import torch
import random
import re
import logging
import numpy as np
from textwrap import dedent
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbot")


has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"
if has_cuda:
    logger.info(f"CUDA is available! Using {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available. Using CPU instead.")


MODEL_PATH = "models/tinyllama_finetuned/final_model"
DB_PATH = "data/course_content.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/course_embeddings.index"
TRAINING_DATA_PATH = "data/training_data.json"
TRAINING_QUESTIONS_PATH = "data/training_questions.json"


logger.info("Loading language model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16,  
    device_map="auto"
)
logger.info(f"Model loaded successfully on {device}")


logger.info("Loading embedding model and FAISS index")
retrieval_model = SentenceTransformer(EMBEDDING_MODEL)
if has_cuda:
    retrieval_model = retrieval_model.to(device)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
logger.info("Embeddings and index loaded")


training_questions_cache = None
training_data_cache = None



def load_training_data():
    global training_data_cache
    if training_data_cache is not None:
        return training_data_cache
    try:
        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items from training data file")
        training_data_cache = data
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading training data: {e}")
        training_data_cache = []
        return []



def load_training_questions():
    global training_questions_cache
    if training_questions_cache is not None:
        return training_questions_cache
    try:
        with open(TRAINING_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from training questions file")
        if questions and len(questions) > 0:
            sample_topics = [q.get('topic', 'Unknown') for q in questions[:5]]
            logger.info(f"Sample topics from questions file: {sample_topics}")
        training_questions_cache = questions
        return questions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading training questions: {e}")
        training_questions_cache = []
        return []



def clean_output(output_text):
    instruction_patterns = [
        r'REQUIREMENTS:.*?(?=\d+\.|\Z)',
        r'FORMAT:.*?(?=\d+\.|\Z)',
        r'TASK:.*?(?=\d+\.|\Z)',
        r'IMPORTANT:.*?(?=\d+\.|\Z)',
        r'YOUR QUESTION[S]?:.*?(?=\d+\.|\Z)',
        r'Your answer:.*?(?=\d+\.|\Z)',
        r'DO NOT.*?(?=\d+\.|\Z)',
        r'Instructions for.*?(?=\d+\.|\Z)',
        r'Here is.*?(?=\d+\.|\Z)',
        r'QUESTION TYPES.*?(?=\d+\.|\Z)',
        r'EXAMPLE.*?(?=\d+\.|\Z)',
        r'BEGIN QUESTIONS:.*?(?=\d+\.|\Z)',
        r'REFERENCE INFORMATION.*?(?=\d+\.|\Z)',
        r'EACH QUESTION MUST:.*?(?=\d+\.|\Z)',
        r'QUESTIONS:.*?(?=\d+\.|\Z)',
        r'EXAMPLES OF GOOD QUESTIONS:.*?(?=\d+\.|\Z)',
        r'SPECIFIC EXAMPLES:.*?(?=\d+\.|\Z)',
        r'EXAMPLE QUESTIONS.*?(?=\d+\.|\Z)',
        r'GENERATE NEW QUESTIONS:.*?(?=\d+\.|\Z)',
        r'USE THIS REFERENCE INFORMATION.*?(?=\d+\.|\Z)'
    ]
    for pattern in instruction_patterns:
        output_text = re.sub(pattern, '', output_text, flags=re.DOTALL|re.IGNORECASE)
    output_text = re.sub(r'^\s*\d+\.\s*', '', output_text, flags=re.MULTILINE)
    output_text = re.sub(r'(\d+\.|•|\*|\-).*?:', '', output_text)
    output_text = re.sub(r'(?i)(must be|should be|is|are)(\s\w+){0,3}\s(original|your own|unique)(\screation)?', '', output_text)
    output_text = re.sub(r'(?i)(here are|the following|below are)(\s\w+){0,3}\s(questions|examples)', '', output_text)
    if output_text and not output_text[-1] in ['.', '!', '?']:
        last_period = max(output_text.rfind('.'), output_text.rfind('!'), output_text.rfind('?'))
        if last_period > len(output_text) * 0.5:
            output_text = output_text[:last_period+1]
    output_text = re.sub(r'\n{3,}', '\n\n', output_text)
    output_text = re.sub(r'\s{2,}', ' ', output_text)
    return output_text.strip()



def retrieve_relevant_text(query, max_results=5, similarity_threshold=0.65):
    logger.info(f"Retrieving context for: {query}")
    results = []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic, detailed_content FROM course_content WHERE LOWER(topic) = ?", (query.lower(),))
    db_exact_matches = cursor.fetchall()
    if not db_exact_matches:
        cursor.execute("SELECT topic, detailed_content FROM course_content WHERE topic LIKE ?", (f"%{query}%",))
        db_results = cursor.fetchall()
    else:
        db_results = db_exact_matches
    for topic, content in db_results:
        if content and content.strip():
            results.append({
                'source': 'database',
                'topic': topic,
                'content': content.strip(),
                'score': 1.0
            })
    conn.close()
    if len(results) < max_results:
        query_embedding = retrieval_model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, k=max_results * 2)
        training_data = load_training_data()
        for i, idx in enumerate(indices[0]):
            if idx < len(training_data):
                similarity = 1.0 - distances[0][i]
                if similarity >= similarity_threshold:
                    content = training_data[idx].get('detailed_content', '')
                    topic = training_data[idx].get('topic', 'Unknown')
                    if content and content.strip():
                        if not any(r['content'] == content.strip() for r in results):
                            results.append({
                                'source': 'training_data',
                                'topic': topic,
                                'content': content.strip(),
                                'score': similarity
                            })
    results.sort(key=lambda x: x['score'], reverse=True)
    results = results[:max_results]
    if not results:
        logger.warning(f"No relevant content found for: {query}")
        return None
    for i, result in enumerate(results):
        logger.info(f"Retrieved content {i+1}: Topic={result['topic']}, Source={result['source']}, Score={result['score']:.2f}")
    combined_text = ""
    for result in results:
        combined_text += f"Topic: {result['topic']}\n{result['content']}\n\n"
    return combined_text.strip()



def normalize_topic(topic):
    abbreviations = {
        "nn": "neural networks",
        "neural network": "neural networks",
        "neural net": "neural networks",
        "cnn": "convolutional neural networks",
        "conv net": "convolutional neural networks",
        "convolutional net": "convolutional neural networks",
        "convolutional network": "convolutional neural networks",
        "rnn": "recurrent neural networks",
        "recurrent net": "recurrent neural networks",
        "nlp": "natural language processing",
        "rl": "reinforcement learning",
        "dl": "deep learning",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "gan": "generative adversarial networks",
        "svm": "support vector machines",
        "knn": "k-nearest neighbors",
        "k-nn": "k-nearest neighbors",
        "k nearest neighbor": "k-nearest neighbors",
        "pca": "principal component analysis",
    }
    normalized = topic.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    if normalized in abbreviations:
        normalized = abbreviations[normalized]
    normalized_words = normalized.split()
    for i, word in enumerate(normalized_words):
        if word in abbreviations:
            normalized_words[i] = abbreviations[word]
    normalized = " ".join(normalized_words)
    return normalized



def find_questions_from_file(topic, count=10, threshold=0.6):
    questions = load_training_questions()
    if not questions:
        logger.warning("No training questions available")
        return []
    topic_normalized = normalize_topic(topic)
    logger.info(f"Looking for questions on topic: '{topic}' (normalized: '{topic_normalized}')")
    topic_words = set(topic_normalized.split())
    exact_matches = []
    for item in questions:
        if 'topic' not in item or 'question' not in item:
            continue
        item_topic = item.get('topic', '').lower()
        normalized_item_topic = normalize_topic(item_topic)
        if normalized_item_topic == topic_normalized:
            exact_matches.append(item)
            logger.debug(f"Exact match: {item_topic} == {topic_normalized}")
    logger.info(f"Found {len(exact_matches)} exact matches for '{topic}'")
    if len(exact_matches) >= count:
        logger.info(f"Using {count} exact matches from training file")
        return random.sample(exact_matches, count)
    partial_matches = []
    for item in questions:
        if 'topic' not in item or 'question' not in item:
            continue
        if item in exact_matches:
            continue
        item_topic = normalize_topic(item['topic'])
        if topic_normalized in item_topic:
            partial_matches.append(item)
            logger.debug(f"Partial match (contains): {topic_normalized} in {item_topic}")
            continue
        if item_topic in topic_normalized:
            partial_matches.append(item)
            logger.debug(f"Partial match (contained in): {item_topic} in {topic_normalized}")
            continue
        item_topic_words = set(item_topic.split())
        if topic_words and item_topic_words and not topic_words.isdisjoint(item_topic_words):
            overlap = len(topic_words.intersection(item_topic_words))
            union = len(topic_words.union(item_topic_words))
            overlap_ratio = overlap / union
            if overlap_ratio >= threshold:
                partial_matches.append(item)
                logger.debug(f"Word overlap match: {item_topic} with {topic_normalized}, ratio: {overlap_ratio:.2f}")
    all_matches = exact_matches.copy()
    for item in partial_matches:
        if item not in all_matches:
            all_matches.append(item)
    logger.info(f"Found {len(exact_matches)} exact matches and {len(partial_matches)} partial matches for '{topic}'")
    if all_matches:
        sample_size = min(count, len(all_matches))
        logger.info(f"Using {sample_size} questions from training file ({len(exact_matches)} exact, {len(partial_matches)} partial)")
        return random.sample(all_matches, sample_size)
    logger.warning(f"No matching questions found for topic: {topic}")
    return []



def validate_question(question, topic):
    
    if not isinstance(question, str) or not question.strip():
        return False
    
    
    if not question.endswith('?'):
        return False
    
    
    word_count = len(question.split())
    if word_count < 3 or word_count > 30:
        return False
    
    
    question_content = re.sub(r'[^a-zA-Z0-9\s]', '', question).strip()
    if not question_content:
        return False
    
    
    if "[" in question and "]" in question:
        return False
        
    
    placeholder_terms = ["aspect", "mechanism", "concept", "placeholder", "example", "fill in"]
    if any(term in question.lower() for term in placeholder_terms):
        
        if any(f"the {term}" in question.lower() for term in placeholder_terms):
            return False
    
    
    if '_____' in question or '____' in question or '___' in question:
        return False

    
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'whose', 'whom', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
    has_question_word = any(question.lower().startswith(word) for word in question_words)
    
    
    if not has_question_word and word_count < 5:
        return False
    
    
    if topic and len(topic.strip()) > 0:
        
        topic_lower = topic.lower()
        question_lower = question.lower()
        
        
        topic_terms = set(topic_lower.split())
        if not any(term in question_lower for term in topic_terms):
        
            
            pass
    
    return True

def extract_questions_from_text(generated_text, topic):
    
    logger.info(f"Beginning question extraction from text: {generated_text[:100]}")
    
    
    text = clean_output(generated_text)
    
    
    potential_questions = []
    
    
    question_pattern = re.compile(r'([^.!?]+\?)')
    matches = question_pattern.findall(text)
    for match in matches:
        cleaned = match.strip()
        
        cleaned = re.sub(r'^[\d\.\)\s]+', '', cleaned)  
        cleaned = re.sub(r'^[^\w\s]+\s*', '', cleaned)  
        cleaned = re.sub(r'^Generate\s*[:-]?\s*', '', cleaned, flags=re.IGNORECASE)  
        cleaned = cleaned.strip()
        if cleaned and cleaned.endswith('?'):
            potential_questions.append(cleaned)
    
    logger.info(f"Found {len(potential_questions)} potential questions with first pattern")
    
    
    if not potential_questions:
        
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'do', 'does', 'can']
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            
            line = re.sub(r'^[\d\.\)\s]+', '', line)  
            line = re.sub(r'^[^\w\s]+\s*', '', line)  
            line = re.sub(r'^Generate\s*[:-]?\s*', '', line, flags=re.IGNORECASE)  
            
            
            lower_line = line.lower()
            if any(lower_line.startswith(word) for word in question_words):
                
                if not line.endswith('?'):
                    line += '?'
                potential_questions.append(line)
    
    logger.info(f"After second pattern, found {len(potential_questions)} potential questions")
    
    
    valid_questions = []
    for q in potential_questions:
        
        if len(q.split()) < 3:
            continue
            
        
        if q and q[0].isalpha() and not q[0].isupper():
            q = q[0].upper() + q[1:]
            
        
        if '[' in q and ']' in q:
            continue
            
        
        placeholder_terms = ["aspect", "mechanism", "concept", "placeholder", "example", "fill in", "[blank]"]
        if any(term in q.lower() for term in placeholder_terms):
            continue
            
        valid_questions.append(q)
    
    logger.info(f"After validation, found {len(valid_questions)} valid questions")
    
    
    unique_questions = []
    for q in valid_questions:
        if not any(q.lower() == uq.lower() for uq in unique_questions):
            unique_questions.append(q)
    
    return unique_questions

def generate_questions_with_model(topic, count=1, force_generation=True):
    
    logger.info(f"Generating {count} questions on topic: '{topic}' (force_generation={force_generation})")
    final_questions = []
    
    
    if not force_generation:
        logger.info("Checking for existing questions in database")
        existing_questions = find_questions_from_file(topic, count=count)
        file_questions = [q["question"] for q in existing_questions]
        logger.info(f"Found {len(file_questions)} existing questions in database")
        if file_questions:
            selected_questions = random.sample(file_questions, min(count, len(file_questions)))
            for q in selected_questions:
                if not q.endswith('?'):
                    q += '?'
                if not q[0].isupper() and q[0].isalpha():
                    q = q[0].upper() + q[1:]
                final_questions.append(q)
            if len(final_questions) >= count:
                return final_questions[:count]
    
    
    retrieval_context = retrieve_relevant_text(topic, max_results=2)
    
    
    example_questions = []
    
    
    normalized_topic = normalize_topic(topic)
    training_questions = load_training_questions()
    
    exact_matches = []
    for item in training_questions:
        if 'topic' not in item or 'question' not in item:
            continue
        item_topic = normalize_topic(item.get('topic', '').lower())
        if item_topic == normalized_topic:
            exact_matches.append(item["question"])
    
    
    if len(exact_matches) < 5:
        related_topics = []
        for item in training_questions:
            if 'topic' not in item or 'question' not in item:
                continue
            item_topic = normalize_topic(item.get('topic', '').lower())
            if normalized_topic in item_topic or item_topic in normalized_topic:
                related_topics.append(item["question"])
    
    
    if exact_matches:
        example_questions.extend(random.sample(exact_matches, min(5, len(exact_matches))))
    elif related_topics:
        example_questions.extend(random.sample(related_topics, min(5, len(related_topics))))
    
    
    remaining_needed = count - len(final_questions)
    if remaining_needed <= 0:
        return final_questions
    
    
    prompt = f"""
    Generate {remaining_needed * 10} clear, specific questions about {topic}.
    
    IMPORTANT REQUIREMENTS:
    - Generate ({remaining_needed}) questions that are unique and not duplicates
    -
    - Each question must be numbered (1., 2., etc.)
    - Each question must end with a question mark
    - Questions must be complete sentences
    - Each question should be specific and detailed
    - DO NOT use placeholders or generic terms
    - DO NOT GENERATE SAME QUESTION TWICE
    
    """
    
    
    if example_questions:
        prompt += "EXAMPLE QUESTIONS FROM SIMILAR TOPICS:\n"
        for i, q in enumerate(example_questions, 1):
            prompt += f"{i}. {q}\n"
        prompt += "\n"
    
    
    if retrieval_context:
        prompt += f"USE THIS REFERENCE INFORMATION TO CREATE SPECIFIC QUESTIONS:\n{retrieval_context}\n\n"
    
    prompt += "YOUR QUESTIONS:\n"
    
    
    logger.info(f"Generating questions with prompt: {prompt[:200]}...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=500,  
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    logger.info(f"Raw generated text: {generated[:300]}")
    
    
    extracted_questions = extract_questions_from_text(generated, topic)
    logger.info(f"Extracted {len(extracted_questions)} questions from first generation")
    
    
    for q in extracted_questions:
        if q.lower() not in [fq.lower() for fq in final_questions]:
            final_questions.append(q)
            if len(final_questions) >= count:
                return final_questions[:count]
    
    
    remaining_needed = count - len(final_questions)
    if remaining_needed > 0:
        logger.info(f"Still need {remaining_needed} questions, trying with higher temperature")
        
        
        prompt = f"""
        Generate {remaining_needed * 3} different, creative questions about {topic}.
        
        REQUIREMENTS:
        - Each question must be complete and end with a question mark
        - Questions should cover different aspects of {topic}
        - Questions should be insightful and challenging
        - DO NOT repeat these questions: {', '.join(final_questions)}
        - DO NOT GENERATE SAME QUESTION TWICE
        
        """
        
        if retrieval_context:
            prompt += f"REFERENCE INFORMATION:\n{retrieval_context}\n\n"
            
        prompt += "GENERATE NEW QUESTIONS:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.95,  
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        logger.info(f"Raw generated text (second attempt): {generated[:300]}...")
        
        
        more_questions = extract_questions_from_text(generated, topic)
        logger.info(f"Extracted {len(more_questions)} questions from second generation")
        
        
        for q in more_questions:
            if q.lower() not in [fq.lower() for fq in final_questions]:
                final_questions.append(q)
                if len(final_questions) >= count:
                    return final_questions[:count]
    
    
    remaining_needed = count - len(final_questions)
    if remaining_needed > 0:
        logger.info(f"Making final attempt to generate {remaining_needed} more questions")
        
        
        question_starters = [
            f"What are the key advantages of {topic}?",
            f"How does {topic} compare to traditional approaches?",
            f"Why is {topic} important in modern applications?",
            f"What challenges are associated with implementing {topic}?",
            f"How have recent advancements improved {topic}?",
            f"What role does learning rate play in {topic}?",
            f"How does regularization prevent overfitting in {topic}?",
            f"What architectures are most effective for {topic}?",
            f"How can performance be optimized in {topic}?",
            f"What future developments are expected in {topic}?"
        ]
        
        
        random.shuffle(question_starters)
        final_questions.extend(question_starters[:remaining_needed])
        
        logger.info(f"Returning {len(final_questions)} questions to user: {final_questions}")
    
    
    
    if len(final_questions) >= count:
       return random.sample(final_questions, count)
    else:
        return final_questions

def generate_single_question(topic, force_generation=True):
   
    logger.info(f"Generating a single question on topic: '{topic}' (force_generation={force_generation})")
    
    
    if not force_generation:
        existing_questions = find_questions_from_file(topic, count=3)
        if existing_questions:
            question = random.choice(existing_questions)["question"]
            logger.info(f"Using existing question from database: '{question}'")
            if not question.endswith('?'):
                question += '?'
            if not question[0].isupper() and question[0].isalpha():
                question = question[0].upper() + question[1:]
            return question
    
    
    retrieval_context = retrieve_relevant_text(topic, max_results=2)
    
    
    normalized_topic = normalize_topic(topic)
    training_questions = load_training_questions()
    
    example_questions = []
    for item in training_questions:
        if 'topic' not in item or 'question' not in item:
            continue
        item_topic = normalize_topic(item.get('topic', '').lower())
        if item_topic == normalized_topic or normalized_topic in item_topic or item_topic in normalized_topic:
            example_questions.append(item["question"])
    
    
    prompt = f"""
    Create ONE excellent question about {topic}.
    
    REQUIREMENTS:
    - The question must be complete and end with a question mark
    - The question should be specific and detailed
    - DO NOT use placeholders like [aspect] or [concept]
    
    """
    
    
    if example_questions:
        prompt += "EXAMPLE QUESTIONS FROM SIMILAR TOPICS:\n"
        samples = random.sample(example_questions, min(3, len(example_questions)))
        for i, q in enumerate(samples, 1):
            prompt += f"{i}. {q}\n"
        prompt += "\n"
    
    
    if retrieval_context:
        prompt += f"REFERENCE INFORMATION:\n{retrieval_context}\n\n"
    
    prompt += f"YOUR QUESTION ABOUT {topic.upper()}:\n"
    
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    logger.info(f"Raw generated text: {generated}")
    
    
    extracted_questions = extract_questions_from_text(generated, topic)
    
    if extracted_questions:
        return extracted_questions[0]
    
    
    logger.warning("Failed to extract a valid question, trying direct completion approach")
    
    
    direct_prompt = f"Complete this question about {topic}: How does" 
    
    inputs = tokenizer(direct_prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    direct_generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    
    if not direct_generated.endswith('?'):
        direct_generated += '?'
    
    return direct_generated


def get_questions(topic, force_generation=True, count=1):
    if count == 1:
        return generate_single_question(topic, force_generation=force_generation)
    questions = generate_questions_with_model(topic, count=count, force_generation=force_generation)
    return questions



def get_response(user_query):
    query = user_query.strip()
    logger.info(f"Generating response for: {query}")
    retrieved_content = retrieve_relevant_text(query, max_results=3)
    prompt = dedent(f"""
        Task: Provide a detailed and informative answer to the following question.
        
        Question: {query}
    """)
    if retrieved_content:
        prompt += dedent(f"""
        
        Here is relevant information to help answer the question:
        {retrieved_content}
        
        Use this information to inform your answer, but explain in your own words.
        """)
    prompt += dedent("""
        
        Instructions for your answer:
        1. Be thorough and informative
        2. Provide detailed explanations with examples when helpful
        3. Use your own words rather than copying phrases directly
        4. Organize your response with clear structure
        5. If the question is about technical topics, include specific details
        6. Aim for a comprehensive answer (4-6 paragraphs)
        
        Your answer:
    """)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if "Your answer:" in response:
        response = response.split("Your answer:")[-1].strip()
    final_response = clean_output(response)
    if len(final_response.split()) < 50:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if "Your answer:" in response:
            response = response.split("Your answer:")[-1].strip()
        final_response = clean_output(response)
    return final_response