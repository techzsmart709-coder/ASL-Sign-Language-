import json
import os

wlasl_path = "e:/SIGN LANGUAGE/ASL/dataset/WLASL_v0.3.json"
output_path = "e:/SIGN LANGUAGE/ASL/config.py"

if not os.path.exists(wlasl_path):
    print(f"Error: {wlasl_path} not found.")
    exit(1)

with open(wlasl_path, 'r') as f:
    data = json.load(f)

# Sort words by number of instances to get the most common ones
# the JSON structure typically lists objects with 'gloss' and 'instances' arrays
word_counts = []
for entry in data:
    gloss = entry.get('gloss')
    instances = entry.get('instances', [])
    word_counts.append((gloss, len(instances)))

# Sort descending by instance count
word_counts.sort(key=lambda x: x[1], reverse=True)

# Extract top 1000 words
top_1000 = [w[0] for w in word_counts[:1000]]

if len(top_1000) < 1000:
    for i in range(1000 - len(top_1000)):
        top_1000.append(f"filler_word_{i}")

common_words = top_1000[:300]
likely_words = top_1000[300:600]
least_likely = top_1000[600:1000]

with open(output_path, 'w') as f:
    f.write('IMAGE_PATH = "dataset/sample_image.jpg"\n\n')
    f.write('MAX_NUM_HANDS = 1\n')
    f.write('MIN_DETECTION_CONFIDENCE = 0.5\n\n')
    
    # Write Common Words
    f.write('COMMON_WORDS = {\n    ')
    f.write(',\n    '.join([f'"{w}"' for w in common_words]))
    f.write('\n}\n\n')
    
    # Write Likely Words
    f.write('LIKELY_WORDS = {\n    ')
    f.write(',\n    '.join([f'"{w}"' for w in likely_words]))
    f.write('\n}\n\n')
    
    # Write Least Likely Words
    f.write('LEAST_LIKELY_WORDS = {\n    ')
    f.write(',\n    '.join([f'"{w}"' for w in least_likely]))
    f.write('\n}\n\n')
    
    # Write Aggregated
    f.write('ALL_WORDS = COMMON_WORDS | LIKELY_WORDS | LEAST_LIKELY_WORDS\n')

print("Wrote config.py with tiered 1000 keywords.")
