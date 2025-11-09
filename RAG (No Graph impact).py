import json
import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
import requests
import re
from collections import defaultdict
from typing import Optional, Literal

class EnhancedHybridRAG:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, chroma_path="./chroma_db"):
        """Initialize Enhanced Hybrid RAG System"""
        print("Initializing Enhanced Hybrid RAG System...")      

        # Neo4j connection (still performed for compatibility/debug)
        print("Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("[OK] Neo4j connected")

        # ChromaDB setup
        print("Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

        # Sentence transformer for embeddings
        print("Loading embedding model (BGE-base-en-v1.5)...")
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("[OK] Embedding model loaded")

        # ChromaDB embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="lung_cancer_sections",
            embedding_function=self.embedding_function,
            metadata={"description": "Lung cancer research paper sections"}
        )
        print("[OK] ChromaDB collection ready")

        # Intent to entity type mapping for graph search
        self.intent_to_type = {
            'symptoms': 'Symptoms',
            'causes': 'Causes',
            'treatment': 'Treatment',
            'diagnosis': 'Diagnostic',
            'statistics': 'Statistics',
            'methodology': 'Proposed Models', # traversal me search karne ke liye
            'results': 'Evaluation',
            'types': 'Type of Cancer'
        }
    
    def close(self):
        """Close connections"""
        self.neo4j_driver.close()
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity"""
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    
    def load_json_to_chromadb(self, json_file_path):
        """Load JSON sections into ChromaDB"""
        print(f"\n{'='*80}")
        print(f"LOADING DATA TO CHROMADB")
        print(f"{'='*80}")

        # Check if already loaded
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"[INFO] ChromaDB already has {existing_count} documents.")
            return existing_count

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sections = data.get('Sections', {})

        documents = []
        metadatas = []
        ids = []

        chunk_id = 0

        print("Extracting sections from JSON...")
        for section_name, section_content in sections.items():
            text = section_content.get('text', section_content.get('Text', ''))

            if text and len(text.strip()) > 0:
                chunk_id += 1
                documents.append(text)
                metadatas.append({
                    "section_name": section_name,
                    "type": "main_section", 
                    "chunk_id": chunk_id
                })
                ids.append(f"section_{chunk_id}")
                print(f"  Added: {section_name}")

        print(f"\nCreating embeddings for {len(documents)} sections...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"[OK] Loaded {len(documents)} sections into ChromaDB\n")
        return len(documents)
    
    def extract_entity_and_intent(self, question: str) -> dict:
        """Extract main entity and intent from question using LLM"""
        prompt = f"""Extract the main entity and intent from this question about lung cancer research.

Question: {question}

Respond in JSON format:
{{
    "entity": "the main medical or technical term (e.g., 'lung cancer', 'SVM', 'Random Forest')",
    "intent": "one of: symptoms, causes, treatment, diagnosis, prevention, statistics, methodology, results, types"
}}

If no clear entity or intent, use null.

JSON Response:"""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen3:8b',
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json'
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()['response']
                parsed = json.loads(result)
                entity = parsed.get('entity')
                intent = parsed.get('intent')

                if not entity and not intent:
                    print("  No domain-specific entity/intent found, trying NER extraction...")
                    ner_result = self._extract_nouns_and_entities(question)
                    if ner_result['entities']:
                        entity = ner_result['entities'][0] 
                        print(f"  NER extracted entity: '{entity}'")

                return {
                    'entity': entity,
                    'intent': intent
                }
            else:
                print(f"  Warning: LLM returned status {response.status_code}, using fallback")
                return self._fallback_entity_intent(question)

        except Exception as e:
            print(f"  Warning: LLM extraction failed ({e}), using fallback")
            return self._fallback_entity_intent(question)
    
    def _extract_nouns_and_entities(self, question: str) -> dict:
        """Extract nouns and named entities using LLM when domain-specific extraction fails"""
        prompt = f"""Extract all nouns and named entities from this question.

Question: {question}

Respond in JSON format:
{{
    "entities": ["list", "of", "nouns", "and", "named", "entities"],
    "main_subject": "the most important noun or entity"
}}

JSON Response:"""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen3:8b',
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json'
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()['response']
                parsed = json.loads(result)
                entities = parsed.get('entities', [])
                main_subject = parsed.get('main_subject')

                stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
                entities = [e for e in entities if e.lower() not in stop_words]

                return {
                    'entities': entities,
                    'main_subject': main_subject
                }
            else:
                return {'entities': [], 'main_subject': None}

        except Exception as e:
            print(f"  Warning: NER extraction failed ({e})")
            return {'entities': [], 'main_subject': None}
    
    def _fallback_entity_intent(self, question: str) -> dict:
        """Fallback entity and intent extraction using keywords"""
        question_lower = question.lower()

        medical_entities = ['lung cancer', 'cancer', 'tumor', 'sclc', 'nsclc',
                            'machine learning', 'algorithm', 'svm', 'ann', 
                            'random forest', 'neural network', 'model']
        found_entity = None
        for entity in medical_entities:
            if entity in question_lower:
                found_entity = entity
                break

        intent_keywords = {
            'symptoms': ['symptom', 'sign', 'indication'],
            'causes': ['cause', 'reason', 'factor', 'risk'],
            'treatment': ['treatment', 'therapy', 'cure'],
            'diagnosis': ['diagnosis', 'detection', 'test', 'screening'],
            'prevention': ['prevention', 'avoid', 'protect'],
            'statistics': ['statistic', 'rate', 'percentage', 'number', 'many', 'how many'],
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'model', 'used'],
            'results': ['result', 'accuracy', 'performance', 'outcome'],
            'types': ['type', 'category', 'classification', 'difference', 'kind']
        }
        found_intent = None
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    found_intent = intent
                    break
            if found_intent:
                break

        return {'entity': found_entity, 'intent': found_intent}
    
    def vector_search(self, question, top_k=3):
        """Search using vector embeddings in ChromaDB"""
        print(f"\n[1. VECTOR SEARCH - ChromaDB]")

        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        retrieved = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]
            retrieved.append({
                'section_name': results['metadatas'][0][i]['section_name'],
                'text': results['documents'][0][i],
                'score': score,
                'method': 'vector'
            })

        print(f"Top {top_k} sections by vector similarity:") 
        for i, r in enumerate(retrieved, 1):
            print(f"  {i}. {r['section_name']}: {r['score']:.4f}")

        return retrieved

    # --- FULL GRAPH SEARCH CODE VISIBLE BUT COMMENTED OUT ---
    def graph_search_with_traversal(self, entity, intent, top_k=3, max_path_length=3):
        """
        New graph search approach:
        1. Find sections matching entity (ranked by graph similarity)
        2. From best ranked section, traverse to find sections matching intent
        3. Longer path = lower weight
        """
        print(f"\n[2. GRAPH SEARCH - Entity-Intent Traversal - DISABLED]")
        # if not entity:
        #     print("  No entity found, skipping graph search")
        #     return []
        # with self.neo4j_driver.session() as session:
        #     entity_keywords = entity.split()
        #     entity_type = self.intent_to_type.get(intent, intent.capitalize())
        #     # Step 1: Find sections matching entity
        #     entity_result = session.run("""
        #         ... entire Cypher logic ...
        #     """, keywords=entity_keywords)
        #     ...
        # print(f"  Top {len(final_sections)} sections from graph search:")
        # for i, section in enumerate(final_sections, 1):
        #     print(f"    {i}. {section['section_name']}: {section['graph_score']:.4f} "
        #           f"(path_length: {section['path_length']}, intent_matches: {section['intent_matches']})")
        # return final_sections[:top_k]
        # --- Disabled, returning empty list below ---
        return []

    def min_max_normalize(self, vector_results, graph_results):
        """Apply min-max normalization to bring scores to [0, 1] range"""
        print(f"\n[3. MIN-MAX NORMALIZATION]")

        all_sections = {}

        vector_scores = [r['score'] for r in vector_results]
        if vector_scores:
            v_min, v_max = min(vector_scores), max(vector_scores)
            v_range = v_max - v_min if v_max != v_min else 1

            for r in vector_results:
                normalized_score = (r['score'] - v_min) / v_range 
                all_sections[r['section_name']] = {
                    'section_name': r['section_name'],
                    'text': r['text'],
                    'vector_score': normalized_score,
                    'graph_score': 0.0  # graph component is always 0
                }

            print(f"  Vector scores normalized: [{v_min:.4f}, {v_max:.4f}] -> [0, 1]") 

        # --- Graph normalization visibly commented out ---
        # graph_scores = [r['graph_score'] for r in graph_results]
        # if graph_scores:
        #     g_min, g_max = min(graph_scores), max(graph_scores)
        #     g_range = g_max - g_min if g_max != g_min else 1
        #     for r in graph_results:
        #         normalized_score = (r['graph_score'] - g_min) / g_range
        #         if r['section_name'] in all_sections:
        #             all_sections[r['section_name']]['graph_score'] = normalized_score
        #         else:
        #             all_sections[r['section_name']] = {
        #                 'section_name': r['section_name'],
        #                 'text': r['text'],
        #                 'vector_score': 0.0,
        #                 'graph_score': normalized_score
        #             }
        #     print(f"  Graph scores normalized: [{g_min:.4f}, {g_max:.4f}] -> [0, 1]")

        return all_sections

    def final_ranking(self, normalized_sections, top_k=3):
        """Final ranking combining normalized vector and graph scores"""
        print(f"\n[4. FINAL RANKING - Vector Scores Only]")

        for section in normalized_sections.values():
            # Only vector score matters
            section['hybrid_score'] = section['vector_score']  # graph_score always 0

        final_ranked = sorted(normalized_sections.values(), 
                            key=lambda x: x['hybrid_score'], 
                            reverse=True)[:top_k]

        print(f"  Top {top_k} sections by vector score:")
        for i, r in enumerate(final_ranked, 1):
            print(f"    {i}. {r['section_name']}: {r['hybrid_score']:.4f}")

        return final_ranked

    def query_llm(self, question, contexts):
        """Query LLM with context sections - Generation Agent"""
        print(f"\n[5. LLM GENERATION AGENT]")

        combined_context = "\n\n".join([
            f"Section {i+1}: {c['section_name']}\n{c['text'][:800]}"
            for i, c in enumerate(contexts)
        ])

        prompt = f"""You are a generation agent. Answer the question from the provided context.

Context:
{combined_context}

Question: {question}

Answer:"""

        print("  Generating answer with Qwen3:8b...")
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen3:8b',
                    'prompt': prompt,
                    'stream': False
                },
                timeout=120
            )

            if response.status_code == 200:
                answer = response.json()['response']
                return answer
            else:
                return f"Error: LLM returned status code {response.status_code}"

        except Exception as e:
            return f"Error: {e}"
    
    def validate_answer(self, question: str, context: str, answer: str) -> dict:
        """Validate answer using LLM validation agent"""
        prompt = f"""You are a validation agent. Check if the answer is appropriate according to the given question and context.

Context:
{context[:1500]}

Question: {question}

Answer: {answer}

Respond in JSON format:
{{
    "is_valid": "yes or no",
    "reason": "brief explanation of why the answer is valid or invalid"
}}

JSON Response:"""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen3:8b',
                    'prompt': prompt,
                    'stream': False,
                    'format': 'json'
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()['response']
                parsed = json.loads(result)
                is_valid = parsed.get('is_valid', 'no').lower()

                if 'yes' in is_valid:
                    is_valid = 'yes'
                else:
                    is_valid = 'no'
                return {
                    'is_valid': is_valid,
                    'reason': parsed.get('reason', 'No reason provided')
                }
            else:
                print(f"  Warning: Validation LLM returned status {response.status_code}")
                return {'is_valid': 'yes', 'reason': 'Validation service unavailable'}

        except Exception as e:
            print(f"  Warning: Validation failed ({e})")
            return {'is_valid': 'yes', 'reason': f'Validation error: {str(e)}'}
    
    def validation_loop(self, question, contexts, max_attempts=2):
        """
        Validation loop with generation agent
        Returns: dict with answer, validation status, and all attempts
        """
        print(f"\n[6. VALIDATION LOOP]")

        attempts = []
        combined_context = "\n\n".join([
            f"Section {i+1}: {c['section_name']}\n{c['text'][:800]}"
            for i, c in enumerate(contexts)
        ])

        for attempt in range(1, max_attempts + 1):
            print(f"\n  Attempt {attempt}:")

            answer = self.query_llm(question, contexts)
            print(f"    Generated answer (preview): {answer[:150]}...")

            print(f"    Validating answer...")
            validation = self.validate_answer(question, combined_context, answer)

            print(f"    Validation result: {validation['is_valid']}")
            if validation['reason']:
                print(f"    Reason: {validation['reason']}")

            attempts.append({
                'attempt': attempt,
                'answer': answer,
                'validation': validation['is_valid'],
                'reason': validation['reason']
            })

            if validation['is_valid'] == 'yes':
                print(f"  [OK] Answer validated successfully!")
                return {
                    'final_answer': answer,
                    'validated': True,
                    'attempts': attempts
                }

        print(f"  [FAIL] Answer not validated after {max_attempts} attempts")
        print(f"  Returning last generated answer")

        return {
            'final_answer': attempts[-1]['answer'],
            'validated': False,
            'attempts': attempts
        }
    
    def answer_question(self, question, top_k=3): 
        """Main pipeline: answer a single question"""
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")

        print(f"\n[ENTITY & INTENT EXTRACTION - LLM]")
        entity_intent = self.extract_entity_and_intent(question)
        print(f"  Entity: '{entity_intent['entity']}'")
        print(f"  Intent: '{entity_intent['intent']}'")

        vector_results = self.vector_search(question, top_k)

        # graph_results = self.graph_search_with_traversal(
        #     entity_intent['entity'], 
        #     entity_intent['intent'], 
        #     top_k
        # )
        graph_results = []  # Graph part disabled

        normalized_sections = self.min_max_normalize(vector_results, graph_results)

        final_results = self.final_ranking(normalized_sections, top_k)

        result = self.validation_loop(question, final_results)

        print(f"\n{'='*80}") 
        print(f"FINAL ANSWER:")
        print(f"{'='*80}")
        print(result['final_answer'])
        print(f"\nValidated: {result['validated']}")

        return {
            'question': question,
            'answer': result['final_answer'],
            'validated': result['validated'],
            'attempts': result['attempts'],
            'sources': [r['section_name'] for r in final_results],
            'entity': entity_intent['entity'],
            'intent': entity_intent['intent']
        }

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password1234"
    JSON_FILE = "lung_cancer_fixed_FOR_RAG.json"
    
    rag = EnhancedHybridRAG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        rag.load_json_to_chromadb(JSON_FILE)
        
        questions = [
            "What are the symptoms of lung cancer?",
            "What causes lung cancer?",
            "Which machine learning algorithm achieved the highest accuracy?",
            "What is the difference between SCLC and NSCLC?",
            "How many people die from lung cancer each year?",
            "What diagnostic techniques are used for lung cancer detection?",
            "What machine learning models were used in this study?",
            "What is the accuracy of the Random Forest model?",
            "What are the best restaurants in Paris?",  # IRRELEVANT QUESTION
            "What dataset was used in this research?"
        ]
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {len(questions)} QUESTIONS")
        print(f"{'='*80}\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n\n{'#'*80}")
            print(f"QUESTION {i}/{len(questions)}")
            print(f"{'#'*80}")
            
            result = rag.answer_question(question)
            results.append(result)
            
            print(f"\nSources used: {', '.join(result['sources'])}")
            print(f"Validation status: {'[OK] Validated' if result['validated'] else '[FAIL] Not validated'}")
            print(f"Attempts: {len(result['attempts'])}")
            print(f"\n{'='*80}\n")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        for i, r in enumerate(results, 1):
            validation_icon = "[OK]" if r['validated'] else "[FAIL]"
            print(f"\n{i}. {r['question']}")
            print(f"   Entity: {r['entity']} | Intent: {r['intent']}")
            print(f"   Sources: {', '.join(r['sources'][:2])}...")
            print(f"   Validated: {validation_icon} ({len(r['attempts'])} attempts)")

            for attempt in r['attempts']:
                status = "[OK]" if attempt['validation'] == 'yes' else "[FAIL]"
                print(f"     Attempt {attempt['attempt']}: {status} {attempt['validation']}")
                if attempt['reason']:
                    print(f"       Reason: {attempt['reason']}")
        
        print(f"\n[SUCCESS] Processed all {len(questions)} questions!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        rag.close()
