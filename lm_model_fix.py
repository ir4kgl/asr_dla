upper_lm_path = 'data/upper_3-gram.pruned.1e-7.arpa'
lm_path = 'data/3-gram.pruned.1e-7.arpa'

with open(upper_lm_path, 'r') as f_upper:
    with open(lm_path, 'w') as f_lower:
        for line in f_upper:
            f_lower.write(line.lower())