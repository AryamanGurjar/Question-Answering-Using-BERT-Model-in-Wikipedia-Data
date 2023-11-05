from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from datasets import load_dataset

app = Flask(__name__)

# Load the "wiki_qa" dataset from Hugging Face API
dataset = load_dataset("wiki_qa", split="train[:100]")  # Load the first 100 rows

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask_question', methods=['POST'])
def ask_question():
    # Retrieve the question from the request
    data = request.get_json()
    question = data['question']

    # Initialize variables to keep track of the best answer and its confidence
    best_answer = None
    best_confidence = 0.0

    for example in dataset:
        context = example['answer']

        # Tokenize the question and context
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

        # Get start and end scores from the model
        start_scores, end_scores = model(**inputs, return_dict=False)

        # Find the answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Retrieve and format the answer
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
        answer = ' '.join(tokens[start_index:end_index + 1])

        # Calculate the confidence score (sum of start and end scores)
        confidence = start_scores[0][start_index] + end_scores[0][end_index]

        # Check if this answer has higher confidence than the best answer
        if confidence > best_confidence:
            best_answer = answer
            best_confidence = confidence

    # Render the index.html template with the predicted answer
    print(best_answer)
    return render_template('index.html', predicted_answer=best_answer)

if __name__ == '__main__':
    app.run(debug=True)
