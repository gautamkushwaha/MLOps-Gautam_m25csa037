
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pretrained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input and output file paths
input_file = "/home/m25csa037/mlops_final_exam/MLDLOPs-Exam2026/Q1/input.rtf" # Changed to use uploaded input.rtf
output_file = "/home/m25csa037/mlops_final_exam/MLDLOPs-Exam2026/Q1/output.rtf" # This will be the generated translation output

# Read input sentences
with open(input_file, "r", encoding="utf-8") as f:
    sentences = f.readlines()

translated_sentences = []
for sentence in sentences:
    sentence = sentence.strip()
    if not sentence:
        translated_sentences.append("")
        continue
    # Tokenize and translate
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_sentences.append(decoded_output)

# Save translated sentences to output file
with open(output_file, "w", encoding="utf-8") as f:
    for ts in translated_sentences:
        f.write(ts + "\n")

print(f"Translation complete. Translated text saved to {output_file}")