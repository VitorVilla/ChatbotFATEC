import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import requests
from transformers import pipeline

# Baixar os recursos necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Carregar o modelo de linguagem em português do spaCy
nlp = spacy.load("pt_core_news_md")

# Modelo de Q&A da biblioteca Transformers
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

#------------- FUNÇÕES ------------#

# mensagem de boas-vindas
def welcome_message(user_text):
    greetings = ["hello", "hi", "hey", "greetings", "olá", "oi", "oie"]
    for word in word_tokenize(user_text.lower()):
        if word in greetings:
            return "Olá! Como posso te ajudar hoje?"
    return None

# preprocessamento para remover pontuação, stop words e tokenizar
def preprocessamento(texto):
    stop_words = set(stopwords.words('portuguese'))
    texto = texto.lower()  # converte o texto para minúsculas
    texto = texto.translate(str.maketrans('', '', string.punctuation))  # remove pontuação
    doc = nlp(texto)  # processa o texto com spaCy para lematização
    tokens_filtrados = [token.lemma_ for token in doc if token.text not in stop_words]  # remove stop words e aplica lematização
    return " ".join(tokens_filtrados)  # retorna os tokens filtrados como uma string

# carregar base de conhecimento
def carregar_conteudo():
    response = requests.get(url)

    # verificar se a requisição foi bem-sucedida
    if response.status_code == 200:
        return response.text
    else:
        return f"Erro ao acessar o arquivo: {response.status_code}"

# calcular a similaridade usando TF-IDF e cosine similarity
def calcular_similaridade(pergunta, sentencas):
    vectorizer = TfidfVectorizer()
    
    # criar uma lista de strings para TF-IDF a partir da pergunta e das sentenças preprocessadas
    docs = sentencas + [pergunta]  # adiciona a pergunta ao final da lista de sentenças
    tfidf_matrix = vectorizer.fit_transform(docs)
    similaridades = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()  # calcula a similaridade entre a pergunta e as sentenças
    return similaridades

# encontrar as melhores respostas com base em um limiar de similaridade
def answer(user_text, threshold=0.1):
    user_text = preprocessamento(user_text)  # preprocessar a pergunta do usuário
    similaridades = calcular_similaridade(user_text, sentencas_preprocessadas)  # calcular a similaridade com as sentenças preprocessadas

    # encontrar todas as sentenças com similaridade acima do limiar
    respostas_relevantes = [f"* {sentencas_original[i]}" for i, similaridade in enumerate(similaridades) if similaridade >= threshold]

    # se nenhuma sentença for relevante, retornar uma mensagem padrão
    if not respostas_relevantes:
        return "Desculpe, não consegui encontrar uma resposta relevante."

    # else, retornar todas as sentenças relevantes, cada uma em uma nova linha
    return "\n".join(respostas_relevantes)

#-------- BUSCANDO O ARQUIVO --------------#

# URL do arquivo raw no GitHub
url = 'https://raw.githubusercontent.com/thainazanfolin/chatbot-pln/refs/heads/main/curso-desenvolvimento-software.txt'

# baixar o conteúdo do arquivo .txt usando a def criada
texto_fatec = carregar_conteudo()  # carregar o conteúdo na variável que vamos utilizar com o texto sobre o curso

#------- PREPROCESSAMENTO ------------#

# melhorar a segmentação de sentenças com o sent_tokenize do NLTK
sentencas_original = sent_tokenize(texto_fatec)  # dividir em sentenças com NLTK para melhor segmentação
sentencas_preprocessadas = [preprocessamento(sentenca) for sentenca in sentencas_original]  # preprocessar cada sentença

#---------- CHATBOT -----------#

# implementando o loop de interação do chatbot
cont = True
print('Olá! Eu sou um chatbot e vou responder suas perguntas sobre o curso de Desenvolvimento de Software Multiplataforma da FATEC Araras. Escreva SAIR quando quiser encerrar nossa conversa!')

# while cont:
#     user_text = input()  # coletar a entrada do usuário

#     if user_text.lower() != 'sair':
#         # se o texto contém uma saudação (ver função)
#         if welcome_message(user_text) is not None:
#             print('Chatbot: ' + welcome_message(user_text))
#         else:
#             print('Chatbot: \n' + answer(user_text))  # usar a função answer() para responder
#     else:
#         cont = False
#         print('Chatbot: Tchau! Até a próxima!')
