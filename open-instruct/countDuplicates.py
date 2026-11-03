import pandas as pd, json

# Load the data
file_name='/path/to/scratch/data/processed/llama3/siqa/high_p0/100_high_p0_siqa_train_options_answer_p0.json'
df = pd.read_json(file_name, lines=True)

# Group by the column you're checking (e.g., "messages")
# and count how many unique IDs each value has
repeated_values = df.groupby('perplexity')['id'].nunique()

# Filter for values that appear in more than one ID
duplicates = repeated_values[repeated_values > 1]




# Step 1: Count how many unique IDs each "likelihood" value appears in
value_to_ids = df.groupby('perplexity')['id'].nunique()

# Step 2: Filter for values that appear in more than one ID
duplicate_likelihoods = value_to_ids[value_to_ids > 1].index

# Step 3: Filter original dataframe to keep only rows with those duplicate likelihood values
duplicate_rows = df[df['perplexity'].isin(duplicate_likelihoods)]

# Step 4: Group by 'likelihood' to see which IDs share them
ids_with_duplicate_values = duplicate_rows.groupby('perplexity')['id'].unique()

#print("IDs with duplicate 'perplexity' values:")
#print(ids_with_duplicate_values)



# Load the data
file_name = '/path/to/scratch/data/processed/llama3/siqa/high_p0/100_high_p0_siqa_train_options_answer_p0.json'
df = pd.read_json(file_name, lines=True)

# Create combined content from the two message parts
df['combined_content'] = df['messages'].apply(lambda msgs: msgs[0]['content'] +' '+ msgs[1]['content'])

# Find perplexities associated with more than one ID
duplicate_likelihoods = df.groupby('perplexity')['id'].nunique()
duplicate_likelihoods = duplicate_likelihoods[duplicate_likelihoods > 1].index

# Filter rows for those perplexities
duplicate_rows = df[df['perplexity'].isin(duplicate_likelihoods)]

# Group and print results
grouped = duplicate_rows.groupby('perplexity')
match_pair = 0
unmatch_pair = 0
similar_path = '/path/to/scratch/data/processed/llama3/siqa/high_p0/100_high_p0_siqa_train_options_answer_p0_duplicates.json'
f1 = open(similar_path,'w')
for perplexity, group in grouped:
    data_pair = {}
    ids = group['id'].tolist()
    unique_contents = group['combined_content'].unique()
    match = len(unique_contents) == 1
    if match:
        match_pair+=1
        data_pair['content'] = unique_contents[0]
    temp = {}
    for j, id in enumerate(ids):
        temp[str(j)] = id
    data_pair['ids'] = temp
    data_pair['match'] = match
    data_pair['perplexity'] = perplexity
    print(f"Perplexity {perplexity}: Match = {match}, IDs = {ids}")
    if not match:
        unmatch_pair+=1
        print("  Different combined contents:")
        temp_c = {}
        for k, content in enumerate(unique_contents):
            #print("   ---")
            #print(content)
            temp_c[str(k)] = content
        data_pair['content'] = temp_c
    json.dump(data_pair, f1)
    f1.write('\n')

print('match_pair ',match_pair, ' unmatch_pair ',unmatch_pair)

