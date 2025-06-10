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
import pandas as pd
from datetime import timedelta
from textblob import TextBlob


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
        Initialize the RAG system for WhatsApp chats
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.ollama_model = ollama_model
        self.messages: List[ChatMessage] = []
        self.embeddings = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        self.faiss_index = None
        self.emoji_length = 0
        
        # Common emojis mapping for normalization
        self.common_emojis = "😂👀😭🤔😌😉😅🥺😵‍💫🤣🥵😏🫂😆🥲😃😄🙇💀🙂🫣"
        
        # Emoji patterns for better search
        self.emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        
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
        print("Sample current message:", messages[0].content, messages[1].content, messages[2].content, messages[3].content, messages[4].content if messages else "No messages found.")
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
        Create embeddings using multiple techniques for better retrieval
        """
        if len(self.messages) == 0:
            print("No messages to embed. Please parse a chat file first.")
            return
        
        # Prepare texts for embedding
        texts = []
        for msg in self.messages:
            # Create rich context for embedding
            context = f"{msg.sender}: {msg.content}"
            # Add temporal context
            context += f" [Time: {msg.timestamp}]"
            # Add message type context
            context += f" [Type: {msg.message_type}]"
            texts.append(context)
        
        print("Creating sentence embeddings...")
        # Create sentence embeddings
        self.embeddings = self.embedding_model.encode(texts)
        
        print("Creating TF-IDF matrix...")
        # Create TF-IDF matrix for keyword-based search
        simple_texts = [msg.content for msg in self.messages]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(simple_texts)
        
        print("Creating FAISS index...")
        # Create FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Created embeddings for {len(self.messages)} messages!")

    def get_advanced_analytics(self):
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
        hourly_by_sender = df.groupby(['sender', 'hour']).size().unstack(fill_value=0)
        # Ensure all 24 hours are present
        for h in range(24):
            if h not in hourly_by_sender.columns:
                hourly_by_sender[h] = 0
        hourly_by_sender = hourly_by_sender[sorted(hourly_by_sender.columns)]

                

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
            'hourly_by_sender': hourly_by_sender  
        }

    def generate_response(self, prompt: str) -> str:
        try:
            return self.query_ollama(prompt)  # Assuming LangChain-compatible call
        except Exception as e:
            print("LLM response failed:", e)
            return "🤖 AI was too stunned to roast your chat properly."

    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Tuple[ChatMessage, float]]:
        """
        Hybrid search combining semantic and keyword-based retrieval
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
    
    def contextual_search(self, query: str, sender_filter: str = None, 
                         date_filter: str = None, message_type: str = None) -> List[ChatMessage]:
        """
        Advanced search with filters
        """
        filtered_messages = self.messages.copy()
        
        # Apply filters
        if sender_filter:
            filtered_messages = [msg for msg in filtered_messages 
                               if sender_filter.lower() in msg.sender.lower()]
        
        if date_filter:
            filtered_messages = [msg for msg in filtered_messages 
                               if date_filter in msg.timestamp]
        
        if message_type:
            filtered_messages = [msg for msg in filtered_messages 
                               if msg.message_type == message_type]
        
        # Perform search on filtered messages
        if len(filtered_messages) == 0:
            return []
        
        # Create temporary embeddings for filtered messages
        texts = [f"{msg.sender}: {msg.content}" for msg in filtered_messages]
        temp_embeddings = self.embedding_model.encode(texts)
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, temp_embeddings).flatten()
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        return [filtered_messages[i] for i in sorted_indices[:10]]
    
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
    
    def answer_question(self, question: str, use_context: bool = True) -> str:
        """
        Answer questions using RAG
        """
        if len(self.messages) == 0:
            return "No chat data loaded. Please parse a chat file first."
        
        # Search for relevant messages
        relevant_messages = self.hybrid_search(question, top_k=20)
        
        if not relevant_messages:
            return "💬 Answer: I couldn't find any relevant messages for your query. Try rephrasing your question or asking about different topics from the chat."
        
        # Prepare context from relevant messages
        context = "Relevant chat messages:\n\n"
        for i, (msg, score) in enumerate(relevant_messages, 1):
            context += f"{i}. [{msg.timestamp}] {msg.sender}: {msg.original_content}\n"
        
        context += f"\n\nTotal messages in chat: {len(self.messages)}"
        context += f"\nSearch confidence: {relevant_messages[0][1]:.2f}"
        
        # Create prompt for Ollama
        prompt = f"""You are an AI assistant analyzing a WhatsApp chat conversation between friends/family who communicate in a mix of Tamil and English (code-switching). 

CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
🎯 **Response Guidelines:**
- Answer based on the provided chat messages
- Be conversational and natural, like a friend who knows this chat group
- Handle Tamil-English code-switching naturally (don't translate unless asked)
- Include relevant direct quotes from messages when helpful
- If asked about emotions/relationships, read between the lines from the chat context
- Use emojis appropriately to match the chat's tone

📝 **Response Format:**
- Start with a direct answer to the question
- Support with specific evidence from the messages
- Include relevant quotes in their original language mix
- End with additional insights if relevant

🚫 **Don't:**
- Make up information not in the chat
- Translate everything to English unless specifically asked
- Be overly formal (match the casual chat tone)

💡 **Special Handling:**
- If question is about specific people, focus on their messages
- If about events, provide timeline from timestamps
- If about relationships/emotions, analyze communication patterns

ANSWER:"""

        return self.query_ollama(prompt)
    
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

# Example usage
def main():
    # Initialize the RAG system
    rag = WhatsAppChatRAG()
    
    # Parse WhatsApp chat file
    chat_file = r"C:\Users\akhsh\Desktop\LLM_Training\chat.txt"  # Replace with your file path
    messages = rag.parse_whatsapp_chat(chat_file)
    
    if not messages:
        print("Failed to parse chat file. Please check the file format.")
        return
    
    # Create embeddings
    rag.create_embeddings()
    
    # Get chat statistics
    stats = rag.get_chat_statistics()
    print("\n📊 Chat Statistics:")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Message types: {dict(stats['message_types'])}")
    print(f"Top senders: {[f'{name} ({count})' for name, count in stats['top_senders']]}")
    print(f"Total emojis used: {stats['total_emojis']}")
    
    # Interactive question-answering
    print("\n🤖 Chat RAG System Ready!")
    print("Ask questions about your WhatsApp chat. Type 'quit' to exit.")
    
    while True:
        question = input("\n❓ Your Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching and generating answer...")
        answer = rag.answer_question(question)
        print(f"\n💬 Answer: {answer}")

if __name__ == "__main__":
    main()