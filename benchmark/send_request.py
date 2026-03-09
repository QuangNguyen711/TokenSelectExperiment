import time
import openai
import csv
import sys
from transformers import AutoTokenizer

MODEL_PATH="Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

port=62726
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

key="Quang cười haha"
fake_key="Quang cười hihi"

# Get output filename from command line argument
if len(sys.argv) > 1:
    output_csv = sys.argv[1]
else:
    output_csv = input("Enter output CSV filename (e.g., results.csv): ")

# Define the range of n_repeat values to test
# From 50 to 10000 with reasonable intervals
n_repeat_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

# Store results
results = []

print("Running experiments from n_repeat=50 to n_repeat=10000...")
print("=" * 100)

for n_repeat in n_repeat_values:
    prompt = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * n_repeat + f"The pass key is {key}. Remember it. {fake_key} is not the pass key. " + "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * n_repeat + "What is the pass key?"
    
    input_tokens = len(tokenizer.encode(prompt))
    
    print(f"\nRunning n_repeat={n_repeat}, input_tokens={input_tokens}...")
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL_PATH,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=100,  # Limit response length to prevent infinite generation
        stream=True,
    )
    
    first_token_received = False
    ttft = None
    answer = ""
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            if not first_token_received:
                ttft = time.time() - start_time
                first_token_received = True
            answer += chunk.choices[0].delta.content
    
    total_time = time.time() - start_time
    
    results.append({
        'n_repeat': n_repeat,
        'input_tokens': input_tokens,
        'ttft': ttft,
        'answer': answer.strip(),
        'total_time': total_time
    })
    
    print(f"  TTFT: {ttft:.3f}s")
    print(f"  Answer: {answer.strip()}")
    print(f"  Total time: {total_time:.3f}s")

# Print results table
print("\n" + "=" * 100)
print("RESULTS TABLE")
print("=" * 100)
print(f"{'n_repeat':<10} {'Input Tokens':<15} {'TTFT (s)':<12} {'Answer':<50}")
print("-" * 100)

for result in results:
    answer_short = result['answer'][:47] + "..." if len(result['answer']) > 50 else result['answer']
    print(f"{result['n_repeat']:<10} {result['input_tokens']:<15} {result['ttft']:<12.3f} {answer_short:<50}")

print("=" * 100)

# Export to CSV
print(f"\nExporting results to {output_csv}...")
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['n_repeat', 'input_tokens', 'ttft', 'total_time', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"✓ Results exported to {output_csv}")