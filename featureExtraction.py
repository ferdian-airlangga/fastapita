from PyPDF2 import PdfReader
import pandas as pd
import re
import nltk
from deep_translator import GoogleTranslator
import tempfile
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import os
import time
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import requests
from io import BytesIO
import PyPDF2
from docx import Document
import json
from pdf2image import convert_from_bytes



###############################OPERATOR FUNCTION####################################
#TOKENIZE
def tokenize_1_words(text):
    # Tokenisasi teks menjadi array berisi kata-kata
    tokens = nltk.word_tokenize(text)
    
    return tokens

def tokenize_2_words(text):
    # Hilangkan whitespace yang berlebihan
    text = re.sub('\s+', ' ', text).strip()
    
    # Pisahkan teks menjadi array berisi kata-kata
    words = text.split()
    
    # Buat array baru untuk token
    tokens = []
    
    # Loop over kata-kata dan tambahkan ke token setiap 2 kata
    for i in range(0, len(words)-1, 2):
        chunk = words[i:i+2]
        for j in range(len(chunk)):
            token = ' '.join(chunk[j:])
            tokens.append(token)
    
    # Jika ada kata yang tersisa, tambahkan ke token terakhir
    if len(words) % 2 == 1:
        tokens[-1] += ' ' + words[-1]
    
    return tokens


def tokenize_3_words(text):
    # Hilangkan whitespace yang berlebihan
    text = re.sub('\s+', ' ', text).strip()
    
    # Pisahkan teks menjadi array berisi kata-kata
    words = text.split()
    
    # Buat array baru untuk token
    tokens = []
    
    # Loop over kata-kata dan tambahkan ke token setiap 3 kata
    for i in range(0, len(words), 3):
        chunk = words[i:i+3]
        for j in range(len(chunk)):
            token = ' '.join(chunk[j:])
            tokens.append(token)
    
    return tokens

#SKILLS EXTRACTION
def skills_extraction(tokenize_text):    
    df = pd.read_csv('skills_updated.csv')
    df['skills']=df['skills'].str.lower()
    extracted_skills=[]
    skills=df.stack().tolist()
    for token in tokenize_text:
        if token in skills:
            extracted_skills.append(token)
    extracted_skills=set(extracted_skills)
    return extracted_skills

#EXTRACTOR SKILLS FROM RESUME TEXT
def extractor_skills_resume_text (text) :
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    return skills

###############################MAIN FUNCTION####################################

#LINK TO TEXT
def extract_text_from_doc(url):
        response = requests.get(url)
        file = response.content
        file = BytesIO(response.content)
        text=""
        reader = PdfReader(file)
        page_number=len(reader.pages)
        for i in range (page_number):
                page = reader.pages[i]  
                page_text = page.extract_text()
                page_text = GoogleTranslator(source='auto', target='en').translate(page_text)
                text = text+page_text
        text = text.replace("\n", " ")
        text=text.lower()
        return text

#LINK TO TXT (OCR)
def extract_text_from_doc_ocr(url) : 
        response = requests.get(url)
        pdf_bytes = response.content
        images = convert_from_bytes(pdf_bytes)
        text = ''
        for i, image in enumerate(images):
            image = image.convert("L")
            txt = pytesseract.image_to_string(image).encode("utf-8")
            txt = txt.decode('utf8')
            txt = GoogleTranslator(source='auto', target='en').translate(txt)
            text=text+txt
        text = text.replace("¢", " ")
        text = text.replace("\n", " ")
        text = text.lower()
        return text


#EXTRACTOR SKILLS FROM RESUME FILE OCR AND PyPDF2
def extractor_skills_from_resume_balanced (url) :
    text=extract_text_from_doc(url)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    if(len(skills)==0):
        text=extract_text_from_doc_ocr(url)
        tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
        skills = skills_extraction(tokenize_words)        
    return skills

#EXTRACTOR SKILLS FROM RESUME FILE OCR ONLY
def extractor_skills_from_resume_ocr (url) :
    text=extract_text_from_doc_ocr(url)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    return skills

#EXTRACTOR SKILLS FROM RESUME FILE PyPDF2 ONLY
def extractor_skills_from_resume (url) :
    text=extract_text_from_doc(url)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    return skills

###############################ADDITIONAL FUNCTION####################################
#EXTRACT SKILLS TO DATAFRAME 
def extract_skills_df (df) :
    df['cvFile'] = df['cvFile'].apply(lambda x: extractor_skills_from_resume_balanced(x))
#    df['cvFile'] = df['cvFile'].apply(lambda x: ' '.join([word for word in x]))

    df.rename(columns={"cvFile": "skills"},inplace=True)

    return df

#SIMILARITY CALCULATOR
def hitung_skor(set1, set2):
    print(set1)
    print(set2)
    z = set1.intersection(set2)
    print(z)
    similarity = len(z)/len(set1)
    similarity = similarity*100
    similarity = round(similarity, 2)    
    return similarity
#RESUME SCORING
def resume_scoring (dataset,jobdes) :
    jobdes = GoogleTranslator(source='auto', target='en').translate(jobdes)
    jobdes = jobdes.lower()
    jobdes = extractor_skills_resume_text(jobdes)
    dataset.loc[len(dataset)] = pd.Series({'_id': 'jobdes', 'skills': jobdes})
    dataset['skills_concated'] = dataset['skills'].apply(lambda x: ' '.join([word for word in x]))
    print(dataset)
    #Vectorization
    v = TfidfVectorizer()
    dataset_vectorized = v.fit_transform(dataset['skills_concated'])
    dataset_vectorized=pd.DataFrame.sparse.from_spmatrix(dataset_vectorized)

    #Modelling
    true_k = 2
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(dataset_vectorized)
    labels=model.labels_
    cluster_labels = model.labels_
    dataset['cluster'] = cluster_labels
    dataset['cluster'] = dataset['cluster']+1
    jobdes_cluster = dataset.loc[dataset['_id'] == 'jobdes']
    print(jobdes_cluster)
    jobdes_cluster = jobdes_cluster.iloc[0,3]

    #Loop Dataframe
    dataset['score']=dataset['cluster']
    for index, row in dataset.iterrows():
        if(dataset.at[index, 'cluster'] == jobdes_cluster):
            dataset.at[index, 'cluster'] = hitung_skor(jobdes,dataset.at[index, 'skills'])
        else :
            dataset.at[index, 'cluster'] = 0
    print(dataset)
    dataset.rename(columns={"cluster": "score", "score": "cluster"},inplace=True)
    dataset = dataset[dataset['_id'] != 'jobdes']
    dataset = dataset.drop(columns=['skills_concated','cluster'])
    dataset.rename(columns={'_id':'id'}, inplace=True)
    return dataset


#JOBDESC EXTRACTOR
def jobdesc_extractor(file) :
    if file.endswith('.pdf') :
        filePath = file
        doc = convert_from_path(filePath)
        path, fileName = os.path.split(filePath)
        fileBaseName, fileExtension = os.path.splitext(fileName)
        text=""
        for page_number, page_data in enumerate(doc):
            txt = pytesseract.image_to_string(page_data).encode("utf-8")
            txt = txt.decode('utf8')
            txt = GoogleTranslator(source='auto', target='en').translate(txt)
            text=text+txt
        text = text.split("KUALIFIKASI")
        kualifikasi = text[1]
        job_description = text [0]
        job_description = job_description.split("DESKRIPSI PEKERJAAN")
        job_description = job_description [1]
        return [job_description,kualifikasi]
    elif file.endswith('.docx') :
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, 'output.pdf')
        subprocess.run(['unoconv', '-f', 'pdf', '-o', output_file, file], check=True)
        file=output_file
        text=""
        filePath = file
        doc = convert_from_path(filePath)
        path, fileName = os.path.split(filePath)
        fileBaseName, fileExtension = os.path.splitext(fileName)
        text=""
        for page_number, page_data in enumerate(doc):
            txt = pytesseract.image_to_string(page_data).encode("utf-8")
            txt = txt.decode('utf8')
            txt = GoogleTranslator(source='auto', target='en').translate(txt)            
            text=text+txt
        text = text.split("KUALIFIKASI")
        kualifikasi = text[1]
        job_description = text [0]
        job_description = job_description.split("DESKRIPSI PEKERJAAN")
        job_description = job_description [1]
        return [job_description,kualifikasi]
    else :
        print('Sistem hanya menerima file PDF dan DOCX')