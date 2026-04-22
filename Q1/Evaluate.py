import sacrebleu
import os

# Load translated output and reference text
output_file = "output.txt" # This is the generated translation
reference_file = "output.rtf" # This is the user-provided reference

with open(output_file, "r", encoding="utf-8") as f:
    hypothesis = [line.strip() for line in f.readlines()]

with open(reference_file, "r", encoding="utf-8") as f:
    # sacrebleu expects a list of references, even if only one
    # Each reference itself is a list of segments
    references = [[line.strip() for line in f.readlines()]]

# Compute BLEU score
bleu = sacrebleu.corpus_bleu(hypothesis, references)

print(f"BLEU score: {bleu.score:.2f}")

# Get the first translated statement from the output
if hypothesis:
    print(f"\nFirst translated statement from output.txt:\n{hypothesis[0]}")
else:
    print("\noutput.txt is empty or could not be read.")