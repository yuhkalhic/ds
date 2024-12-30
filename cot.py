import json
import re
from zhipuai import ZhipuAI
import time
from tqdm import tqdm
import collections
import math


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

def extract_code(response):
    """Extract code from model response."""

    answer_match = re.search(r'<ANSWER>:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    

    code_block = re.search(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    

    single_line = re.search(r'`(.*?)`', response)
    if single_line:
        return single_line.group(1).strip()
    

    return response.strip().split('\n')[0]

def get_model_response(client, prompt):
    """Get response from ZhipuAI model with few-shot CoT."""
    prompt_template = f"""{FEW_SHOT_EXAMPLES}

Task: {prompt}
Let's approach this step by step:
1. Understand exactly what code needs to be written
2. Break down the implementation into its core components
3. Write the most concise and efficient code solution
Important: Your answer must follow the format:
<ANSWER>: code_here
Only provide the exact code needed, no explanations or backticks in the answer."""

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Python coding assistant. Always provide ONLY the exact code needed, no explanations. Your response must be in the format <ANSWER>: code_here"
                },
                {"role": "user", "content": prompt_template}
            ],
        )
        return extract_code(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error getting model response: {e}")
        return ""

def _get_ngrams(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment."""
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
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

def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate strings."""
    if not candidate:
        return 0.0
    

    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    
    references = [[reference_tokens]]  
    translations = [candidate_tokens]
    

    return compute_bleu(references, translations, max_order=4, smooth=True)

def main():
    api_key = "c23f7a066e71ef6d0a1372ee83fa2297.tKirlJjM2UMqp2IQ" 
    client = ZhipuAI(api_key=api_key)
    
    test_data_path = r"data/test_data.jsonl"
    test_data = load_test_data(test_data_path)
    
    results = []
    total_bleu = 0.0
    
    for item in tqdm(test_data, desc="Processing test cases"):

        prompt = item['rewritten_intent'] if item.get('rewritten_intent') else item['intent']
        model_response = get_model_response(client, prompt)
        
        bleu_score = calculate_bleu_score(item['snippet'], model_response)
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
    
    output_file = 'baseline_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'average_bleu': avg_bleu,
            'results': results
        }, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()