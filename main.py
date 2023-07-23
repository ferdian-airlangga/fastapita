from fastapi import FastAPI, UploadFile, File
import uvicorn
import featureExtraction
import pandas as pd
import requests
import PyPDF2
import io
from fastapi.middleware.cors import CORSMiddleware
import json
from mailer import kirim_email


app = FastAPI()

backend_endpoint = "http://ec2-3-87-22-2.compute-1.amazonaws.com:5000"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/resume_scoring")
async def resume_scoring(positionId: str, token_value : str):    
    
    print("start scoring")

    payload = {
        "id" : positionId
    }
    
    header = {'Authorization': token_value}
    response=requests.get(backend_endpoint+"/api/position/get-one-position",params=payload,headers=header)
    json_data = response.json()
    jobdesc = json_data["description"]+" "+json_data["qualification"]
    
    response=requests.get(backend_endpoint+"/api/candidate/get-all-candidate",headers=header)
    json_data = response.json()
    df = pd.DataFrame(json_data['candidates'])
    df = df.loc[df['position'] == positionId]
    df = df[['_id', 'cvFile']]
    df = featureExtraction.extract_skills_df (df)
    df = featureExtraction.resume_scoring(df,jobdesc)
    df = df.to_json(orient='records')
    df = json.loads(df)
    payload = {"scores" : df}
    print(json.dumps(payload,indent=2))
    response=requests.put(backend_endpoint+"/api/candidate/score-candidate",json=payload,headers=header)
    return payload

@app.post("/jobdesc_reader")
async def jobdesc_reader(file: UploadFile = File(ext=[".docx",".pdf"])):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    file=file.filename
    jobdes = featureExtraction.jobdesc_extractor(file)
    return {
            "job_description" : jobdes [0],
            "qualification" : jobdes[1]
    }

@app.post("/mailer")
async def mailer(email_recipient : str,nama_kandidat : str,posisi_dilamar : str):    
    try:
      kirim_email(email_recipient,nama_kandidat,posisi_dilamar)
      return {'message' : 'Email terkirim'}
    except Exception as e:
      print(e)
      return {'message' : 'SMTP server connection error'}
    

if __name__ == "__main__":
    uvicorn.run('main:app',reload=True)