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

#DOC TO TEXT
def extract_text_from_doc(file):
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

#DOC TO TXT (OCR)
def extract_text_from_doc_ocr(file) :
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
        text = text.replace("¢", " ")
        text = text.replace("\n", " ")
        text = text.lower()
        return text


#EXTRACTOR SKILLS FROM RESUME FILE OCR AND PyPDF2
def extractor_skills_from_resume_balanced (file) :
    text=extract_text_from_doc(file)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    if(len(skills)==0):
        text=extract_text_from_doc_ocr(file)
        tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
        skills = skills_extraction(tokenize_words)        
    return skills

#EXTRACTOR SKILLS FROM RESUME FILE OCR ONLY
def extractor_skills_from_resume_ocr (file) :
    text=extract_text_from_doc_ocr(file)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    return skills

#EXTRACTOR SKILLS FROM RESUME FILE PyPDF2 ONLY
def extractor_skills_from_resume (file) :
    text=extract_text_from_doc(file)
    tokenize_words=tokenize_1_words(text)+tokenize_2_words(text)+tokenize_3_words(text)
    skills = skills_extraction(tokenize_words)
    return skills

###############################ADDITIONAL FUNCTION####################################
#EXTRACT SKILLS TO DATAFRAME 
def extract_skills_df (df) :
    df['cvFile'] = df['cvFile'].apply(lambda x: get_pdf_file_and_title(x))
    df['cvFile'] = df['cvFile'].apply(lambda x: extractor_skills_from_resume(x))
    df['cvFile'] = df['cvFile'].apply(lambda x: ' '.join([word for word in x]))
    df.rename(columns={"cvFile": "skills"},inplace=True)

    return df

#SIMILARITY CALCULATOR
def calculate_similarity(sentence1, sentence2):
    # Tokenisasi kalimat menjadi kata-kata
    tokens = [sentence1, sentence2]
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokens)
    
    # Menghitung similarity menggunakan cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return (round(similarity[0][0]*100, 2))

#RESUME SCORING
def resume_scoring (dataset,jobdes) :
    jobdes = GoogleTranslator(source='auto', target='en').translate(jobdes)
    jobdes = jobdes.lower()
    jobdes = extractor_skills_resume_text(jobdes)
    jobdes = " ".join([word for word in jobdes])
    dataset.loc[len(dataset)] = pd.Series({'_id': 'jobdes', 'skills': jobdes})

    #Vectorization
    v = TfidfVectorizer()
    dataset_vectorized = v.fit_transform(dataset['skills'])
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
    jobdes_cluster = jobdes_cluster.iloc[-1,2] 

    #Loop Dataframe
    dataset['score']=dataset['cluster']
    for index, row in dataset.iterrows():
        if(dataset.at[index, 'cluster'] == jobdes_cluster):
            dataset.at[index, 'cluster'] = calculate_similarity(jobdes,dataset.at[index, 'skills'])
        else :
            dataset.at[index, 'cluster'] = 0
    dataset.rename(columns={"cluster": "score", "score": "cluster"},inplace=True)
    dataset = dataset[dataset['_id'] != 'jobdes']
    dataset = dataset.drop(columns=['skills','cluster'])
    return dataset.to_json(orient='records')

#LINK TO DOC
def get_pdf_file_and_title(url):
    response = requests.get(url, stream=True)
    pdf_data = BytesIO(response.content)
    content_disposition = response.headers.get('content-disposition')
    
    if content_disposition:
        filename = content_disposition.split('filename=')[1]
        filename = filename.strip('"\'')

    else:
        filename = os.path.basename(url)
    
    return pdf_data

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
        return text
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