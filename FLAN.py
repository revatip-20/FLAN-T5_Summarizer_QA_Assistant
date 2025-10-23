import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#import from hugging face transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#AutoTokenizer : a tokenizer loader that automatically picks the right tokenizer for the model you choose
#AutoModelForSeq2SeqLM : a pre trained model loader for Sequence to Sequence Language Models

#choose a instruction-tuned model from hugging face model hub
model_name = "google/flan-t5-small"

#a lightweight model suitable for testing and small scale tasks

print(f"FLAN-T5 Summarizer_Q&A_assistant {model_name} loading...")

#load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

#load the sequence to sequence language model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#function takes a text prompt (string) uses a FLAN-T5 model to generate a response

def run_flan(prompt: str,max_new_tokens: int = 128)->str:
    #tokenisation
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True)

    #generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

#function used for summarisation tasks

def summarize_text(text: str)-> str:
    prompt = f"summarize: {text}"
    return run_flan(prompt,max_new_tokens=160)

#function used to load the contents from local file

def load_context(path:str = "context.txt")-> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    except FileNotFoundError:
        return "No context file found. Please provide a valid context.txt file."
    
#function ask FLAN to answer using only the given context

def answer_from_context(question: str, context: str)-> str:
    if not context.strip():
        return "No context available to answer the question."
    
    prompt =(
        "You are a helpful assistant. Answer the question ONLY using the context.\n"
        "If the answer is not in the context, reply exactly : Not found.\n\n"
        f"Context: \n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    return run_flan(prompt,max_new_tokens=120)

def main():
    print("-------------------------------")
    print("FLAN-T5 Summarizer and Q&A Assistant")
    print("-------------------------------\n")

    print("1. Summarize the data")
    print("2.Questions & answers over local context.txt")
    print("0.Exit")
    print("-------------------------------")

    while True:
        choice = input("Enter your choice (1/2/0): ").strip()

        if choice == "0":
            print("Thank you for using the FLAN-T5 assistant. Goodbye!")
            break

        elif choice == "1":
            print("You have selected Summarization option.")
            print("Please enter the text you want to summarize. End with a blank line:")

            lines = []
            while True:
                line = input()
                
                if not line.strip():
                    break
                lines.append(line)

            text = "\n".join(lines).strip()

            if not text:
                print("No text provided for summarization. Please try again.")
                continue

            print("\nGenerating summary...\n")
            print(summarize_text(text))

        elif choice == "2":
            ctx = load_context("context.txt")

            if not ctx.strip():
                print("No context available in context.txt. Please provide a valid context file.")
                continue

            q = input("Enter your question: ").strip()

            if not q:
                print("No question provided. Please try again.")
                continue

            print("\nGenerating answer...\n")
            print(answer_from_context(q, ctx))
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")

if __name__ == "__main__":
    main()  