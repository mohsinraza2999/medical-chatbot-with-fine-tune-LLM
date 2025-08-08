import string
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return ''.join(text)

def chat():
    model_dir = "LLama-2-medical-chatbot"
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)

    while True:
        print("="*50)
        prompt = input("Write your Description (or type 'exit'):\n")
        if prompt.lower() == "exit":
            print("="*50)
            print(" " * 20, "**Thanks for Using**")
            print("="*50)
            break
        result = pipe(f"<s>[INST] {preprocess_text(prompt)} [/INST]")
        question, response = result[0]['generated_text'].split('[/INST]')
        print("User Prompt:", question.strip())
        print("Model Response:", response.strip())

if __name__ == "__main__":
    chat()