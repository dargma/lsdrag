# Bounded search→read interaction probe

After repeated stopping-policy transfer failures, test a different hypothesis:
forcing search then read can reveal useful coordinated behavior that a single
search action followed by H3 misses. This is target discovery, not another
stopping-head fit, learned model result or independent benchmark.

Choose12 questions by SHA256(20261086:+qid) from old72fit with both initial search
branches already measured and read_chunk legal at their next state. Eligibility
uses action availability only; do not select by F1/EM/cost or development failures.
For each question run keyword_search→read_chunk and semantic_search→read_chunk,
then H3 greedy.24new episodes,600seconds after model load. Native tool schemas,
argument generation and reader unchanged. H3model/heads frozen. Every forced step
explicitly tagged; save original H3 probabilities separately from forced ones.
Unavailable second action fails the probe rather than silently changing schedule.

Compare each new path to its saved same-first-search/H3-suffix path, official F1,
EM and generated cost, paired correct/incorrect categories. Record H3-full-episode
reference separately; no best-of-two policy score. Evidence to expand: >=3distinct
questions with strictly higher F1 and nondecreasing EM versus their same-first
reference, plus >=1 pair with positive answer quality preserved and fewer tokens.
This is a diagnostic gate, not replacement for original +5pp/<=70% goal. Report
all costs and negative pairs. No hyperparameter/seed sweep or reserved80 use.
