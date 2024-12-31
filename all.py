
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
import os
import subprocess
import tempfile

FEW_SHOT_EXAMPLES = """Example 1:
Task: get item's position in a list
Let's approach this step by step:
1. We need to find the indices where the value 1 appears in testlist
2. We can use enumerate to get both index and value while iterating
3. Use list comprehension to collect all matching indices
<ANSWER>: [i for (i, x) in enumerate(testlist) if (x == 1)]

Example 2:
Task: How to sort a list of objects based on an attribute of the objects?
Let's approach this step by step:
1. We need to sort the list using a key function
2. The key function (cmpfun) will extract the attribute to sort by
3. We want to sort in descending order (reverse=True)
<ANSWER>: ut.sort(key=cmpfun, reverse=True)

Example 3:
Task: accessing python dictionary
Let's approach this step by step:
1. We need to access a nested dictionary value
2. First get the first element using index 0
3. Then access the 'from_user' key
<ANSWER>: result[0]['from_user']
"""

class CodeRetriever:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.snippets = []
        self.intents = []
    
    def load_documents(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.snippets.append(data['snippet'])
                self.intents.append(data['intent'])
        
        vectors = self.encoder.encode(self.intents, show_progress_bar=True)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        self.index.add(vectors.astype(np.float32))
    
    def retrieve(self, query, k=5):
        if not query:
            return []
        query_vector = self.encoder.encode([str(query)])
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        return [{'intent': self.intents[idx], 'snippet': self.snippets[idx]} for idx in indices[0]]

def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=True):
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
            precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
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

def extract_code(response):
    answer_match = re.search(r'<ANSWER>:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    code_block = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    
    single_line = re.search(r'`(.*?)`', response)
    if single_line:
        return single_line.group(1).strip()
    
    return response.strip()

def get_initial_response(client, intent, retrieved_examples):
    rag_examples = "\n".join([
        f"Example {i+4}:\nTask: {ex['intent']}\nLet's approach this step by step:\n1. Analyze the example code\n2. Extract key implementation details\n<ANSWER>: {ex['snippet']}" 
        for i, ex in enumerate(retrieved_examples)
    ])
    
    examples_text = FEW_SHOT_EXAMPLES + "\n" + rag_examples
    
    prompt = f"""{examples_text}

Task: {intent}
Let's approach this step by step:
1. Understand exactly what code needs to be written
2. Break down the implementation into its core components
3. Write the most concise and efficient code solution
Important: Your answer must follow the format:
<ANSWER>: code_here"""

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "You are a Python coding assistant. Always provide ONLY the exact code needed, no explanations. Your response must be in the format <ANSWER>: code_here"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return extract_code(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting model response: {e}")
        return ""

def get_repair_method(client, code, test_result):
    if test_result == "passed":
        return ""
        
    system_prompt = """You are a code analysis expert. Analyze the code and test results to:
1. Identify the root cause of the error
2. Explain what needs to be fixed
3. Provide a clear repair strategy
Do not provide the actual fix, only the analysis and repair method."""

    prompt = f"""
Code:
```python
{code}
```
Test Result: {test_result}
Please analyze the error and provide repair guidance."""

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting repair method: {e}")
        return ""

def repair_code(client, code, test_result, repair_method):
    if test_result == "passed":
        return code
        
    system_prompt = """You are a Python coding expert. Given the code, test results, and repair guidance:
1. Fix the code according to the repair method
2. Provide ONLY the corrected code
3. Use proper code formatting with markdown
4. Do not include any explanations"""

    prompt = f"""
Original Code:
```python
{code}
```
Test Result: {test_result}
Repair Method: {repair_method}

Please provide the corrected code."""

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return extract_code(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error repairing code: {e}")
        return code

def execute_python_code(code, timeout=10):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            tmp_file.write(code)

        result = subprocess.run(
            ["python", tmp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        os.unlink(tmp_file_path)

        if result.returncode == 0:
            return "passed", True
        else:
            return result.stderr.strip() or "Execution Failed", False
    except subprocess.TimeoutExpired:
        os.unlink(tmp_file_path)
        return "Timeout Error", False
    except Exception as e:
        os.unlink(tmp_file_path)
        return f"Execution Failed: {e}", False

def tokenize(code):
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

def evaluate_code(reference, generated):
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(generated)
    return compute_bleu([[reference_tokens]], [candidate_tokens], max_order=4, smooth=True)

def load_test_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    api_key = "c23f7a066e71ef6d0a1372ee83fa2297.tKirlJjM2UMqp2IQ"
    client = ZhipuAI(api_key=api_key)

    retriever = CodeRetriever()
    retriever.load_documents("data/code_docs.jsonl")

    test_data_path = "data/test_data.jsonl"
    test_data = load_test_data(test_data_path)

    results = []
    total_bleu = {"initial": 0.0, "final": 0.0}
    
    for item in tqdm(test_data, desc="Processing test cases"):
        query = item['rewritten_intent'] if item.get('rewritten_intent') else item['intent']
        
        # Step 1: Initial code generation with RAG and CoT
        retrieved_examples = retriever.retrieve(query)
        initial_code = get_initial_response(client, query, retrieved_examples)
        
        # Step 2: Execute initial code
        result, passed = execute_python_code(initial_code)
        
        # Step 3: Evaluate initial BLEU score
        initial_bleu = evaluate_code(item['snippet'], initial_code)
        
        # Step 4-7: Repair process (skip if initial_bleu > 0.4)
        if initial_bleu <= 0.4 and not passed:
            repair_method = get_repair_method(client, initial_code, result)
            final_code = repair_code(client, initial_code, result, repair_method)
            final_result, final_passed = execute_python_code(final_code)
            final_bleu = evaluate_code(item['snippet'], final_code)
        else:
            repair_method = ""
            final_code = initial_code
            final_result = result
            final_passed = passed
            final_bleu = initial_bleu
        
        # Store results
        results.append({
            'question_id': item['question_id'],
            'intent': item['intent'],
            'reference': item['snippet'],
            'initial_code': initial_code,
            'initial_result': result,
            'initial_passed': passed,
            'initial_bleu': initial_bleu,
            'repair_method': repair_method,
            'final_code': final_code,
            'final_result': final_result,
            'final_passed': final_passed,
            'final_bleu': final_bleu
        })
        
        total_bleu["initial"] += initial_bleu
        total_bleu["final"] += final_bleu
        
        time.sleep(0.5)

    success = {
        'initial': sum(1 for res in results if res['initial_passed']) / len(results),
        'final': sum(1 for res in results if res['final_passed']) / len(results)
    }
    
    avg_bleu = {
        "initial": total_bleu["initial"] / len(test_data),
        "final": total_bleu["final"] / len(test_data)
    }

    output_data = {
        'success_rate': success,
        'average_bleu': avg_bleu,
        'results': results,
        'metadata': {
            'total_samples': len(test_data),
            'repairs_attempted': len([r for r in results if r['repair_method']]),
            'high_initial_bleu': len([r for r in results if r['initial_bleu'] > 0.4])
        }
    }
    
    output_file = 'combined_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print("\nEvaluation complete!")
    print(f"Initial Success Rate: {success['initial']:.4f}")
    print(f"Final Success Rate: {success['final']:.4f}")
    print(f"Average Initial BLEU score: {avg_bleu['initial']:.4f}")
    print(f"Average Final BLEU score: {avg_bleu['final']:.4f}")
    print(f"Repairs attempted: {output_data['metadata']['repairs_attempted']}")
    print(f"High initial BLEU (>0.4): {output_data['metadata']['high_initial_bleu']}")
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()