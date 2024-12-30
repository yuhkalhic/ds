import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from zhipuai import ZhipuAI
import time
from tqdm import tqdm
import re
import collections
import math

class CodeRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.snippets = []
        self.intents = []
    
    def load_documents(self, file_path):
        """Load code documents and create FAISS index."""
        print("Loading documents...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.snippets.append(data['snippet'])
                self.intents.append(data['intent'])
        

        print("Encoding documents...")
        vectors = self.encoder.encode(self.intents, show_progress_bar=True)
        

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        

        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index.add(vectors.astype(np.float32))
        
        print(f"Indexed {len(self.snippets)} documents")
    
    def retrieve(self, query, k=5):
        """Retrieve top-k similar code snippets."""
        if query is None:
            return []
        query_vector = self.encoder.encode([str(query)])
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        results = []
        for idx in indices[0]:
            results.append({
                'intent': self.intents[idx],
                'snippet': self.snippets[idx]
            })
        return results

def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
    """Computes BLEU score of translated segments against references."""
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0

    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                               possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp

    return bleu

def load_test_data(file_path):
    """Load and parse the JSONL test data."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_code(response):
    """Extract code from model response."""

    code_block = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    

    single_line = re.search(r'`(.*?)`', response)
    if single_line:
        return single_line.group(1).strip()
    

    return response.strip()

def get_model_response(client, intent, retrieved_examples):
    """Get response from ZhipuAI model with retrieved examples."""
    examples_text = "\n".join([
        f"Task: {ex['intent']}\nSolution: {ex['snippet']}" 
        for ex in retrieved_examples
    ])
    
    system_prompt = """You are a Python coding assistant. When given a programming task:
1. Provide ONLY the code solution
2. Use proper code formatting with markdown
3. Do not include any explanations or comments
4. Return a single code snippet that directly solves the task"""

    prompt = f"""Here are some example solutions of similar tasks:

{examples_text}

Now, please solve this task: {intent}"""

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return extract_code(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting model response: {e}")
        return ""

def tokenize(code):
    """Simple tokenization for code."""

    code = ' '.join(code.split())

    tokens = []
    current_token = ''
    for char in code:
        if char.isalnum() or char == '_':
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            if not char.isspace():
                tokens.append(char)
    if current_token:
        tokens.append(current_token)
    return tokens

def main():
    api_key = "c23f7a066e71ef6d0a1372ee83fa2297.tKirlJjM2UMqp2IQ"
    client = ZhipuAI(api_key=api_key)


    retriever = CodeRetriever()
    retriever.load_documents("data/code_docs.jsonl")

    test_data_path = "data/test_data.jsonl"
    test_data = load_test_data(test_data_path)

    results = []
    total_bleu = 0.0

    for item in tqdm(test_data, desc="Processing test cases"):

        query = item['rewritten_intent'] if item.get('rewritten_intent') else item['intent']
        

        retrieved_examples = retriever.retrieve(query)


        model_response = get_model_response(
            client, 
            query,
            retrieved_examples
        )
        
        reference_tokens = tokenize(item['snippet'])
        candidate_tokens = tokenize(model_response)
        
        bleu_score = compute_bleu([[reference_tokens]], [candidate_tokens], max_order=4, smooth=True)
        total_bleu += bleu_score
        
        results.append({
            'question_id': item['question_id'],
            'intent': item['intent'],
            'reference': item['snippet'],
            'generated': model_response,
            'bleu_score': bleu_score
        })
        
        time.sleep(0.5)
    
    avg_bleu = total_bleu / len(test_data)
    

    output_data = {
        'average_bleu': avg_bleu,
        'results': results,
        'metadata': {
            'total_samples': len(test_data),
            'successful_responses': len([r for r in results if r['generated']]),
            'perfect_scores': len([r for r in results if r['bleu_score'] == 1.0]),
            'zero_scores': len([r for r in results if r['bleu_score'] == 0.0])
        }
    }
    
    output_file = 'rag_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Successful responses: {output_data['metadata']['successful_responses']}/{output_data['metadata']['total_samples']}")
    print(f"Perfect scores: {output_data['metadata']['perfect_scores']}")
    print(f"Zero scores: {output_data['metadata']['zero_scores']}")
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()