from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

app = Flask(__name__)

dataset = load_dataset("wiki_qa", split="train[:100]")  # Load the first 100 rows


model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask_question', methods=['POST'])

   

    # Detect the language of the text
    

    
    
def ask_question():
    # Retrieve the question from the request
    data = request.get_json()
    translator = Translator()
    question =data['question']
    lang = translator.detect(question).lang
    
    if lang == 'en':
        # If the text is in English, return it as is
        question
    else:
        # Translate the text to English
        translation = translator.translate(question, dest='en')
        question= translation.text


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['answer'])
    similarity_scores = cosine_similarity(tfidf_vectorizer.transform([question]), tfidf_matrix)
    sorted_indices = similarity_scores.argsort()[0][::-1]
    top_n = 5 
    ba = [dataset['answer'][index] for index in sorted_indices[:top_n]]
    cosine_ans=''
    for  a in enumerate(ba):
        cosine_ans=cosine_ans+a[1]
    

    # Initialize variables to keep track of the best answer and its confidence
    best_answer = None
    best_confidence = 0.0

    
    context = cosine_ans
    text = context
    translation = translator.translate(text, dest=lang)
    text= translation.text

    print("\nFull Description:")
    print(text)
    print()
        # Tokenize the question and context
    encoding = tokenizer.encode_plus(text=question,text_pair=context)

    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
    # inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")

    #     # Get start and end scores from the model
    # start_scores, end_scores = model(**inputs, return_dict=False)

    #     # Find the answer span
    # start_index = torch.argmax(start_scores)
    # end_index = torch.argmax(end_scores)

        # Retrieve and format the answer
    # tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
    start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]),return_dict=False)

    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])
        # Calculate the confidence score (sum of start and end scores)
    corrected_answer = ''

    for word in answer.split():

    #If it's a subword token
       if word[0:2] == '##':
          corrected_answer += word[2:]
       else:
          corrected_answer += ' ' + word
    
    translation = translator.translate(corrected_answer, dest=lang)
    corrected_answer= translation.text
    print("Summarize Answer:")
    print(corrected_answer)
    print()
    # Render the index.html template with the predicted answer
    # print(best_answer)
    return render_template('index.html', predicted_answer=corrected_answer)

if __name__ == '__main__':
    app.run(debug=True)
