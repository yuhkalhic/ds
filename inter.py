import json
import re
from zhipuai import ZhipuAI
import time
from tqdm import tqdm
import collections
import math
import os
import subprocess
import tempfile

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

def load_test_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_code(response):
    code_block = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    
    single_line = re.search(r'`(.*?)`', response)
    if single_line:
        return single_line.group(1).strip()
    
    return response.strip()

def get_model_response(client, prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        )
        return extract_code(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting model response: {e}")
        return ""

def code_generation(client, prompt):
    system_prompt = """You are a Python coding assistant. When given a programming task:
1. Provide ONLY the code solution
2. Use proper code formatting with markdown
3. Do not include any explanations or comments
4. Return a single code snippet that directly solves the task"""
    
    return get_model_response(client, prompt, system_prompt)

def cor_generation(client, code, test_result):
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

    return get_model_response(client, prompt, system_prompt)

def code_repair(client, code, test_result, repair_method):
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

    return get_model_response(client, prompt, system_prompt)

def execute_python_code(code, timeout=10):
    """Executes Python code and returns the result and passed status."""
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


def main():
    api_key = "c23f7a066e71ef6d0a1372ee83fa2297.tKirlJjM2UMqp2IQ"
    client = ZhipuAI(api_key=api_key)

    test_data_path = r"data/test_data.jsonl"
    test_data = load_test_data(test_data_path)

    results = []
    total_bleu = {"initial": 0.0, "final": 0.0}
    
    for item in tqdm(test_data, desc="Processing test cases"):
        prompt = item['rewritten_intent'] if item.get('rewritten_intent') else item['intent']

        # Step 1: Initial Code Generation
        initial_code = code_generation(client, prompt)

        # Step 2: Execute initial code and get result
        result, passed = execute_python_code(initial_code)
         
        # Step 3: Evaluate BLEU score for initial code
        initial_bleu = evaluate_code(item['snippet'], initial_code)
        
        # Step 4: Error analysis
        repair_method = cor_generation(client, initial_code, result)

        # Step 5: Code repair
        final_code = code_repair(client, initial_code, result, repair_method)

        # Step 6: Execute final code and get result
        final_result, final_passed = execute_python_code(final_code)

        # Step 7: Evaluate BLEU score for final code
        final_bleu = evaluate_code(item['snippet'], final_code)

        # Step 8: Store results
        results.append({
            'question_id': item['question_id'],
            'intent': item['intent'],
            'reference': item['snippet'],
            'initial_code': initial_code,
            'initial_result': result,
            'initial_passed': passed,
            'initial_bleu': initial_bleu,
            'repair_method': repair_method if result != "passed" else "",
            'final_code': final_code,
            'final_result': final_result,
            'final_passed': final_passed,
            'final_bleu': final_bleu
        })
        
        total_bleu["initial"] += initial_bleu
        total_bleu["final"] += final_bleu
        
        time.sleep(0.5)

    success = {
        'initial': sum(1 for res in results if res['initial_passed']) / len(results) if results else 0,
        'final': sum(1 for res in results if res['final_passed']) / len(results) if results else 0
    }
    
    avg_bleu = {
        "initial": total_bleu["initial"] / len(test_data),
        "final": total_bleu["final"] / len(test_data)
    }
        

    output_file = 'enhanced_baseline_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'success_rate': success,
            'average_bleu': avg_bleu,
            'results': results
        }, f, indent=2)

    print("\nEvaluation complete!")
    print(f"Initial Success Rate: {success['initial']:.4f}")
    print(f"Final Success Rate: {success['final']:.4f}")
    print(f"Average Initial BLEU score: {avg_bleu['initial']:.4f}")
    print(f"Average Final BLEU score: {avg_bleu['final']:.4f}")
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()