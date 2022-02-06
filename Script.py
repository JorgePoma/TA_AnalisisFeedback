#!/usr/bin/env python
# coding: utf-8

#descargarndo nltk y sus librerias necesarias 

#importar libreria nltk para la deteccion de sentimientos y la generacion de comentarios con openIA

import nltk
import os
import openai
import json
import statistics

#autenticacion con openIA

with open('API-key.json', 'r') as key:
    API_key = json.load(key) 
    openai.api_key = API_key['API-key']
engine_list = openai.Engine.list()

d = {}
d['users'] = []

#generar datos de entrada aleatorios con GPT-3 si el archivo .json aun esta vacio

def generate(add):
    response = openai.Completion.create(
      engine="curie-instruct-beta",
      prompt="generate a json list of 100 users with random name and random comment parameters in Spanish about what they like and dislike about a ticketing bus app{'name':'Alex','comment':'i cant buy my ticket please fix it'},{'name':'Jane','comment':'I like the application because it is very agile'},{'name': 'John','comment':'The app service is perfect'},"+add,
      temperature=0.9,
      max_tokens=15,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    data = response["choices"][0]["text"]
    #print(data)
    return data 

def returnData(value):
    
    final = ""
    f = ""
    for x in range(value):
        final = final + generate(final)
    d['users'].append(final)

    with open('data.json', 'w') as file:
        json.dump(d, file, indent=4)      
          
nltk.download('vader_lexicon')
nltk.download('punkt')

import nltk
nltk.download('vader_lexicon')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


#importando las librerias par el analisis de sentimientos 

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize

entradas = []

positivos = []
neutros = []
negativos = []


#leyendo el archivo .json

with open('data.json', 'r') as fi:
    if fi == {}:
        returnData(20);
    else:
        ent = json.load(fi)
        for usuario in ent['users']:
             entradas.append(usuario['comment'])

#metodo para el analizador de sentimientos

def calcSentimiento(listComentarios):
    analizador = SentimentIntensityAnalyzer()
    for comentario in listComentarios:    
        sentences = tokenizer.tokenize(comentario)
        for sentence in sentences:
            #print(sentence)
            scores = analizador.polarity_scores(sentence)
            for key in scores:
                if key == "neg":
                    negativos.append(scores["neg"])
                if key == "neu":
                    neutros.append(scores["neu"])
                if key == "neg":
                    positivos.append(scores["pos"])

calcSentimiento(entradas)

print("\npositivos: " + str(round(statistics.mean(positivos)*100,2))+"%")
print("\nneutros: " + str(round(statistics.mean(neutros)*100,2))+"%")
print("\nnegativos: " + str(round(statistics.mean(negativos)*100,2))+"%")