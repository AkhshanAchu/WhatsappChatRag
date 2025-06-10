import re
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from dataclasses import dataclass
from collections import Counter
import pandas as pd
from datetime import timedelta
from textblob import TextBlob
from datetime import datetime
# BM25 imports - install with: pip install rank-bm25
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


@dataclass
class ChatMessage:
    timestamp: str
    sender: str
    content: str
    original_content: str
    message_type: str  # text, media, sticker
    
class WhatsAppChatRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", ollama_model: str = "deepseek-r1:8b"):
        """
        Initialize the RAG system for WhatsApp chats with BM25 support
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.ollama_model = ollama_model
        self.messages: List[ChatMessage] = []
        self.embeddings = None
        
        # TF-IDF setup
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        
        # BM25 setup
        self.bm25_okapi = None
        self.bm25_plus = None
        self.bm25_l = None
        self.tokenized_messages = []
        
        # FAISS setup
        self.faiss_index = None
        
        # Text preprocessing
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common emojis mapping for normalization
        self.common_emojis = "😂👀😭🤔😌😉😅🥺😵‍💫🤣🥵😏🫂😆🥲😃😄🙇💀🙂🫣"
        
        # Emoji patterns for better search
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        
    def preprocess_text(self, text: str, use_stemming: bool = True) -> List[str]:
        """
        Preprocess text for BM25 (tokenization, lowercasing, stopword removal, stemming)
        """
        # Remove emojis and special characters, keep alphanumeric and spaces
        text = re.sub(self.emoji_pattern, ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Apply stemming if requested
        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
        
    def parse_whatsapp_chat(self, file_path: str) -> List[ChatMessage]:
        """
        Enhanced parser for WhatsApp chat exports
        """
        messages = []
        
        # Multiple patterns to handle different WhatsApp export formats
        regex_pattern = r'\[(\d{2}/\d{2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM))\]([^\[]*)'
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            matches = re.finditer(regex_pattern, content)
            in_items=[]
            for match in matches:
                if match.group(2):
                    group1 = match.group(1).strip().replace('\u202f', '')
                    group2 = match.group(2).strip().split(':',1)
                    in_items.append((group1,group2[0],group2[1].replace('\u200e', '')))
            print(f"Found {len(in_items)} messages in the chat file.")

            for line in in_items:
                timestamp = line[0]
                # Clean sender name (remove emojis from display)
                clean_sender = re.sub(self.emoji_pattern, '', line[1]).strip()
                # Determine message type and clean content
                message_type, cleaned_message = self._classify_message(line[2])
                
                current_message = ChatMessage(
                    timestamp=timestamp,
                    sender=clean_sender,
                    content=cleaned_message['message'],
                    original_content=line[2],
                    message_type=cleaned_message['type']
                )
                matched = True
                
                # Handle multiline messages
                if not matched and current_message:
                    current_message.content += " " + line
                    current_message.original_content += " " + line
                
                # Add the last message
                if current_message:
                    messages.append(current_message)
                
        except Exception as e:
            print(f"Error parsing chat file: {e}")
            return []
        
        self.messages = messages
        print(f"Parsed {len(messages)} messages successfully!")
        print("Sample messages:", [msg.content for msg in messages[:5]] if messages else "No messages found.")
        return messages
    
    def _classify_message(self, message):
        media_patterns = [
        r'Media omitted',
        r'image omitted',
        r'video omitted',
        r'audio omitted',
        r'sticker omitted',
        r'document omitted',
        r'GIF omitted',
        r'Contact card omitted',
        r'This message was deleted.',
        r'This message was edited',
        r'voice call'
        ]

        media_dict = {
        "Media omitted": "Media",
        "image omitted": "Image",
        "video omitted": "Video",
        "audio omitted": "Audio",
        "sticker omitted": "Sticker",
        "document omitted": "Document",
        "GIF omitted": "GIF",
        "Contact card omitted": "Contact Card",
        "This message was deleted.": "Deleted Message",
        "This message was edited": "Edited Message",
        "voice call": "Voice Call"
        }

        common_emojis = "😂👀😭🤔😌😉😅🥺😵‍💫🤣🥵😏🫂😆🥲😃😄🙇💀🙂🫣"
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')

        for pattern in media_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                msg = message[:match.start()].strip() + message[match.end():].strip()
                msg = msg.strip()
                if msg:
                    return "media and text", {"type": media_dict[pattern], "message": msg}
                else:
                    return "media only", {"type": media_dict[pattern], "message": "Media["+media_dict[pattern]+"]"}

        emoji_count = len(self.emoji_pattern.findall(message))
        text_length = len(re.sub(emoji_pattern, '', message).strip())
        
        if emoji_count > 0 and text_length < 5:
            return "emoji", {"type": "Emoji text", "message": message}

        return "text", {"type": "Text", "message": message}
    
    def create_embeddings(self):
        """
        Create embeddings using multiple techniques including BM25
        """
        if len(self.messages) == 0:
            print("No messages to embed. Please parse a chat file first.")
            return
        
        # Prepare texts for embedding
        texts = []
        simple_texts = []
        for msg in self.messages:
            # Create rich context for embedding
            context = f"{msg.sender}: {msg.content}"
            # Add temporal context
            context += f" [Time: {msg.timestamp}]"
            # Add message type context
            context += f" [Type: {msg.message_type}]"
            texts.append(context)
            simple_texts.append(msg.content)
        
        print("Creating sentence embeddings...")
        # Create sentence embeddings
        self.embeddings = self.embedding_model.encode(texts)
        
        print("Creating TF-IDF matrix...")
        # Create TF-IDF matrix for keyword-based search
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(simple_texts)
        
        print("Creating BM25 indices...")
        # Preprocess texts for BM25
        self.tokenized_messages = [self.preprocess_text(text) for text in simple_texts]
        
        # Create different BM25 variants
        self.bm25_okapi = BM25Okapi(self.tokenized_messages)
        self.bm25_plus = BM25Plus(self.tokenized_messages)
        self.bm25_l = BM25L(self.tokenized_messages)
        
        print("Creating FAISS index...")
        # Create FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Created embeddings and BM25 indices for {len(self.messages)} messages!")

    def bm25_search(self, query: str, variant: str = "okapi", top_k: int = 10) -> List[Tuple[ChatMessage, float]]:
        """
        Search using BM25 algorithm
        """
        if self.bm25_okapi is None:
            print("Please create embeddings first!")
            return []
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            print("Query preprocessing resulted in empty tokens")
            return []
        
        # Select BM25 variant
        if variant == "okapi":
            bm25_model = self.bm25_okapi
        elif variant == "plus":
            bm25_model = self.bm25_plus
        elif variant == "l":
            bm25_model = self.bm25_l
        else:
            bm25_model = self.bm25_okapi
        
        # Get BM25 scores
        scores = bm25_model.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                results.append((self.messages[idx], float(scores[idx])))
        
        return results
    
    def advanced_hybrid_search(self, query: str, top_k: int = 5, 
                             semantic_weight: float = 0.4, 
                             tfidf_weight: float = 0.3, 
                             bm25_weight: float = 0.3,
                             bm25_variant: str = "okapi") -> List[Tuple[ChatMessage, float]]:
        """
        Advanced hybrid search combining semantic, TF-IDF, and BM25 search
        """
        if self.embeddings is None or self.tfidf_matrix is None or self.bm25_okapi is None:
            print("Please create embeddings first!")
            return []
        
        # Ensure weights sum to 1
        total_weight = semantic_weight + tfidf_weight + bm25_weight
        semantic_weight /= total_weight
        tfidf_weight /= total_weight
        bm25_weight /= total_weight
        
        # 1. Semantic search using FAISS
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        semantic_scores, semantic_indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k * 3, len(self.messages))
        )
        
        # 2. TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # 3. BM25 search
        query_tokens = self.preprocess_text(query)
        if bm25_variant == "okapi":
            bm25_model = self.bm25_okapi
        elif bm25_variant == "plus":
            bm25_model = self.bm25_plus
        else:
            bm25_model = self.bm25_l
            
        bm25_scores = bm25_model.get_scores(query_tokens) if query_tokens else np.zeros(len(self.messages))
        
        # Normalize all scores to [0, 1] range
        def normalize_scores(scores):
            if len(scores) == 0 or np.max(scores) == 0:
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        # Normalize semantic scores
        semantic_scores_norm = np.zeros(len(self.messages))
        for i, (idx, score) in enumerate(zip(semantic_indices[0], semantic_scores[0])):
            semantic_scores_norm[idx] = score
        semantic_scores_norm = normalize_scores(semantic_scores_norm)
        
        # Normalize TF-IDF and BM25 scores
        tfidf_scores_norm = normalize_scores(tfidf_scores)
        bm25_scores_norm = normalize_scores(bm25_scores)
        
        # Combine scores
        combined_scores = (semantic_weight * semantic_scores_norm + 
                         tfidf_weight * tfidf_scores_norm + 
                         bm25_weight * bm25_scores_norm)
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:  # Only include relevant results
                results.append((self.messages[idx], float(combined_scores[idx])))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Tuple[ChatMessage, float]]:
        """
        Original hybrid search combining semantic and keyword-based retrieval (kept for backward compatibility)
        """
        if self.embeddings is None or self.tfidf_matrix is None:
            print("Please create embeddings first!")
            return []
        
        # Semantic search using FAISS
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        semantic_scores, semantic_indices = self.faiss_index.search(
            query_embedding.astype('float32'), min(top_k * 3, len(self.messages))
        )
        
        # Keyword search using TF-IDF
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for i, (idx, score) in enumerate(zip(semantic_indices[0], semantic_scores[0])):
            combined_scores[idx] = alpha * score
        
        # Add keyword scores
        for idx, score in enumerate(keyword_scores):
            if score > 0:  # Only consider non-zero keyword matches
                if idx in combined_scores:
                    combined_scores[idx] += (1 - alpha) * score
                else:
                    combined_scores[idx] = (1 - alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for idx, score in sorted_results[:top_k]:
            results.append((self.messages[idx], float(score)))
        
        return results
    def contextual_search(self, 
                        query: str = " ", 
                        sender_filter: str = None, 
                        date_filter: str = None, 
                        message_type: str = None,
                        search_method: str = "hybrid",
                        top_k: int = 10,
                        User_query = " "):
        """
        Advanced contextual search with filters, search methods, result control, and optional query suggestions.
        
        Returns:
            Dict with keys:
            - 'results': List of ChatMessage objects
            - 'suggested_queries': Optional List of suggested follow-up queries
        """
        filtered_messages = self.messages.copy()
        filtered_indices = list(range(len(filtered_messages)))

        
        # Apply filters
        if sender_filter:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                            if sender_filter.lower() in msg.sender.lower()]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if date_filter:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                            if date_filter in msg.timestamp]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if message_type:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                            if msg.message_type == message_type]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if len(filtered_messages) == 0:
            print("Running out")
            return {'results': []}
        
        # Perform search based on method
        if search_method == "bm25":
            query_tokens = self.preprocess_text(query)
            if not query_tokens:
                return {'results': []}
            filtered_tokenized = [self.tokenized_messages[idx] for idx in filtered_indices]
            temp_bm25 = BM25Okapi(filtered_tokenized)
            scores = temp_bm25.get_scores(query_tokens)
            sorted_indices = np.argsort(scores)[::-1]
            results = [filtered_messages[i] for i in sorted_indices[:top_k] if scores[i] > 0]

        elif search_method == "semantic":
            texts = [f"{msg.sender}: {msg.content}" for msg in filtered_messages]
            temp_embeddings = self.embedding_model.encode(texts)
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, temp_embeddings).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            results = [filtered_messages[i] for i in sorted_indices[:top_k]]

        else:  # hybrid
            texts = [f"{msg.sender}: {msg.content}" for msg in filtered_messages]
            temp_embeddings = self.embedding_model.encode(texts)
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, temp_embeddings).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            results = [filtered_messages[i] for i in sorted_indices[:top_k]]

        # Generate follow-up query suggestions using context if required
            # Use top 5 messages for suggestion generation
        print(len(results))
        top_context = "\n".join([f"{msg.sender}: {msg.content}" for msg in results])
        prompt = f"""

        General Instructions:
        - You can answer general questions.
        - If asked from context, Answer based only on the provided chat context 
        - If interpretation is asked, from the chat say how you feel
        - Be conversational and natural
        - If asking about specific events or conversations, reference the relevant messages
        - Show the actual message what is conversed in when needed between quotes.
        - If the context doesn't contain enough information, say so honestly
        - Include relevant emojis when appropriate
        - Keep the response informative 


        Given the following context, answer the query asked.

        Context:
        {top_context}

        Query:{User_query}"""

        suggestions = self.query_ollama(prompt).split("\n")
        print(suggestions)
        return {'results': results, 'Report': suggestions}

    def contextual_search_2(self, query: str, sender_filter: str = None, 
                         date_filter: str = None, message_type: str = None,
                         search_method: str = "hybrid") -> List[ChatMessage]:
        """
        Advanced search with filters and different search methods
        """
        filtered_messages = self.messages.copy()
        filtered_indices = list(range(len(self.messages)))
        
        # Apply filters
        if sender_filter:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                           if sender_filter.lower() in msg.sender.lower()]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if date_filter:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                           if date_filter in msg.timestamp]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if message_type:
            filtered_data = [(msg, idx) for msg, idx in zip(filtered_messages, filtered_indices) 
                           if msg.message_type == message_type]
            filtered_messages = [item[0] for item in filtered_data]
            filtered_indices = [item[1] for item in filtered_data]
        
        if len(filtered_messages) == 0:
            return []
        
        # Perform search based on method
        if search_method == "bm25":
            # BM25 search on filtered messages
            query_tokens = self.preprocess_text(query)
            if not query_tokens:
                return []
            
            # Create temporary BM25 index for filtered messages
            filtered_tokenized = [self.tokenized_messages[idx] for idx in filtered_indices]
            temp_bm25 = BM25Okapi(filtered_tokenized)
            scores = temp_bm25.get_scores(query_tokens)
            
            # Sort by scores
            sorted_indices = np.argsort(scores)[::-1]
            return [filtered_messages[i] for i in sorted_indices[:10] if scores[i] > 0]
        
        elif search_method == "semantic":
            # Semantic search on filtered messages
            texts = [f"{msg.sender}: {msg.content}" for msg in filtered_messages]
            temp_embeddings = self.embedding_model.encode(texts)
            query_embedding = self.embedding_model.encode([query])
            
            similarities = cosine_similarity(query_embedding, temp_embeddings).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            
            return {'results': [filtered_messages[i] for i in sorted_indices[:10]],'report':""}
        
        else:  # hybrid
            # Use the advanced hybrid search on filtered messages
            # This is a simplified version for filtered data
            texts = [f"{msg.sender}: {msg.content}" for msg in filtered_messages]
            temp_embeddings = self.embedding_model.encode(texts)
            query_embedding = self.embedding_model.encode([query])
            
            similarities = cosine_similarity(query_embedding, temp_embeddings).flatten()
            sorted_indices = np.argsort(similarities)[::-1]
            
            return {'results': [filtered_messages[i] for i in sorted_indices[:10]],'report':""}



    def get_advanced_analytics(self):
        """
        Get advanced analytics about the chat
        """
        df = pd.DataFrame([msg.__dict__ for msg in self.messages])
        
        try:
            df.to_csv('chat_data.csv', index=False)
        except:
            pass

        df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip('[]'), format='mixed')
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.day_name()
        df['message_length'] = df['original_content'].apply(len)

        # Filter messages only till today
        today = datetime.today().date()
        df = df[df['date'] <= today]

        hourly_by_sender = df.groupby(['sender', 'hour']).size().unstack(fill_value=0)
        for h in range(24):
            if h not in hourly_by_sender.columns:
                hourly_by_sender[h] = 0
        hourly_by_sender = hourly_by_sender[sorted(hourly_by_sender.columns)]

        participant_daily_activity = df.groupby(['date', 'sender']).size().unstack(fill_value=0)

        return {
            'avg_messages_per_day': df.groupby('date').size().mean(),
            'avg_message_length': df['message_length'].mean(),
            'participant_msg_count': df['sender'].value_counts(),
            'participant_avg_length': df.groupby('sender')['message_length'].mean(),
            'participant_avg_per_day': df.groupby(['sender', 'date']).size().groupby('sender').mean(),
            'message_type_distribution': df.groupby(['sender', 'message_type']).size().unstack(fill_value=0),
            'weekday_activity': df['weekday'].value_counts().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0),
            'hourly_activity': df['hour'].value_counts().sort_index(),
            'message_trend_over_time': df.groupby('date').size(),
            'hourly_by_sender': hourly_by_sender,
            'participant_daily_activity': participant_daily_activity
        }


    def generate_response(self, prompt: str) -> str:
        try:
            return self.query_ollama(prompt)
        except Exception as e:
            print("LLM response failed:", e)
            return "🤖 AI was too stunned to respond properly."
    
    def query_ollama(self, prompt: str) -> str:
        """
        Query Ollama model
        """
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.ollama_model,
                    'prompt': prompt,
                    'stream': False
                }
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error querying Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
    
    def answer_question(self, question: str, search_method: str = "advanced_hybrid", top_numbers=25) -> str:
        """
        Answer questions using RAG with multiple search methods
        """
        if len(self.messages) == 0:
            return "No chat data loaded. Please parse a chat file first."
        
        # Search for relevant messages using different methods
        if search_method == "bm25":
            relevant_messages = self.bm25_search(question, top_k=top_numbers)
        elif search_method == "semantic":
            # Use pure semantic search
            query_embedding = self.embedding_model.encode([question])
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_numbers)
            relevant_messages = [(self.messages[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
        elif search_method == "advanced_hybrid":
            relevant_messages = self.advanced_hybrid_search(question, top_k=top_numbers)
        else:  # default hybrid
            relevant_messages = self.hybrid_search(question, top_k=top_numbers)
        
        if not relevant_messages:
            return "💬 Answer: I couldn't find any relevant messages for your query. Try rephrasing your question or asking about different topics from the chat."
        
        # Prepare context from relevant messages
        context = f"Relevant chat messages (using {search_method} search):\n\n"
        for i, (msg, score) in enumerate(relevant_messages, 1):
            context += f"{i}. [{msg.timestamp}] {msg.sender}: {msg.original_content}\n"
        
        context += f"\n\nTotal messages in chat: {len(self.messages)}"
        context += f"\nSearch confidence: {relevant_messages[0][1]:.3f}"
        context += f"\nSearch method used: {search_method}"
        
        # Create prompt for Ollama
        prompt = f"""Based on the following WhatsApp chat messages, answer the user's question accurately and naturally.

Context from chat:
{context}

User Question: {question}

Instructions:
- You can answer general questions.
- If asked from context, Answer based only on the provided chat context 
- If interpretation is asked, from the chat say how you feel
- Be conversational and natural
- If asking about specific events or conversations, reference the relevant messages
- Show the actual message what is conversed in when needed between quotes.
- If the context doesn't contain enough information, say so honestly
- Include relevant emojis when appropriate
- Keep the response informative 

Answer:"""

        return self.query_ollama(prompt)
    
    def compare_search_methods(self, query: str, top_k: int = 5) -> Dict:
        """
        Compare different search methods for the same query
        """
        if self.embeddings is None or self.bm25_okapi is None:
            return {"error": "Please create embeddings first!"}
        
        results = {}
        
        # BM25 Okapi
        bm25_results = self.bm25_search(query, variant="okapi", top_k=top_k)
        results["BM25_Okapi"] = [(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content, score) 
                                for msg, score in bm25_results]
        
        # BM25 Plus
        bm25_plus_results = self.bm25_search(query, variant="plus", top_k=top_k)
        results["BM25_Plus"] = [(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content, score) 
                               for msg, score in bm25_plus_results]
        
        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        semantic_results = [(self.messages[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
        results["Semantic"] = [(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content, score) 
                              for msg, score in semantic_results]
        
        # Advanced hybrid
        hybrid_results = self.advanced_hybrid_search(query, top_k=top_k)
        results["Advanced_Hybrid"] = [(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content, score) 
                                     for msg, score in hybrid_results]
        
        return results
    
    def get_chat_statistics(self) -> Dict:
        """
        Get interesting statistics about the chat
        """
        if len(self.messages) == 0:
            return {}
        
        stats = {
            'total_messages': len(self.messages),
            'message_types': Counter(msg.message_type for msg in self.messages),
            'top_senders': Counter(msg.sender for msg in self.messages).most_common(5),
            'total_emojis': sum(len(self.emoji_pattern.findall(msg.original_content)) for msg in self.messages),
            'avg_message_length': np.mean([len(msg.content) for msg in self.messages if msg.message_type == 'text']) if any(msg.message_type == 'text' for msg in self.messages) else 0,
            'date_range': {
                'first': self.messages[0].timestamp if len(self.messages) > 0 else None,
                'last': self.messages[-1].timestamp if len(self.messages) > 0 else None
            }
        }
        
        return stats

# Example usage and testing
def main():
    # Initialize the RAG system with BM25
    rag = WhatsAppChatRAG()
    
    # Parse WhatsApp chat file
    chat_file = r"C:\Users\akhsh\Desktop\LLM_Training\chat.txt"  # Replace with your file path
    messages = rag.parse_whatsapp_chat(chat_file)
    
    if not messages:
        print("Failed to parse chat file. Please check the file format.")
        return
    
    # Create embeddings (including BM25 indices)
    rag.create_embeddings()
    
    # Get chat statistics
    stats = rag.get_chat_statistics()
    print("\n📊 Chat Statistics:")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Message types: {dict(stats['message_types'])}")
    print(f"Top senders: {[f'{name} ({count})' for name, count in stats['top_senders']]}")
    print(f"Total emojis used: {stats['total_emojis']}")
    
    # Demo different search methods
    print("\n🔍 Testing Different Search Methods:")
    test_query = "what did we talk about yesterday"
    
    print(f"\nQuery: '{test_query}'")
    print("-" * 50)
    
    # Compare search methods
    comparison = rag.compare_search_methods(test_query, top_k=3)
    for method, results in comparison.items():
        print(f"\n{method} Results:")
        for i, (content, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.3f} - {content}")
    
    # Interactive question-answering with method selection
    print("\n🤖 Enhanced Chat RAG System Ready!")
    print("Available search methods: bm25, semantic, advanced_hybrid, hybrid")
    print("Ask questions about your WhatsApp chat. Type 'quit' to exit.")
    print("You can specify search method like: 'method:bm25 your question here'")
    
    while True:
        user_input = input("\n❓ Your Question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Parse method specification
        search_method = "advanced_hybrid"  # default
        question = user_input
        
        if user_input.startswith("method:"):
            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                method_part = parts[0].replace("method:", "")
                if method_part in ["bm25", "semantic", "advanced_hybrid", "hybrid"]:
                    search_method = method_part
                    question = parts[1]
        
        print(f"\n🔍 Searching using {search_method} method...")
        answer = rag.answer_question(question, search_method=search_method)
        print(f"\n💬 Answer: {answer}")

def test_bm25_variants():
    """
    Test different BM25 variants
    """
    rag = WhatsAppChatRAG()
    
    # Parse chat file
    chat_file = r"C:\Users\akhsh\Desktop\LLM_Training\chat.txt"
    messages = rag.parse_whatsapp_chat(chat_file)
    
    if not messages:
        print("No messages found for testing")
        return
    
    rag.create_embeddings()
    
    test_queries = [
        "what time did we meet",
        "funny moment",
        "food restaurant",
        "weekend plans"
    ]
    
    print("🧪 Testing BM25 Variants:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Test different BM25 variants
        for variant in ["okapi", "plus", "l"]:
            results = rag.bm25_search(query, variant=variant, top_k=3)
            print(f"\nBM25 {variant.upper()}:")
            if results:
                for i, (msg, score) in enumerate(results, 1):
                    content_preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                    print(f"  {i}. Score: {score:.3f} - {content_preview}")
            else:
                print("  No results found")

if __name__ == "__main__":
    # You can run either main() for interactive mode or test_bm25_variants() for testing
    main()
    # Uncomment the line below to run BM25 variant testing
    # test_bm25_variants()