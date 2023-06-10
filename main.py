from fastapi import FastAPI, UploadFile, File
import uvicorn
import featureExtraction
import pandas as pd
import requests
import PyPDF2
import io
from fastapi.middleware.cors import CORSMiddleware
import json


app = FastAPI()

backend_endpoint = "https://9441-182-253-194-76.ngrok-free.app"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/resume_scoring")
async def resume_scoring(positionId: str, token_value : str, companyId : str):    
    
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
    
    df = pd.DataFrame(json_data)
    df = df.loc[df['position'] == positionId]
    df = df[['_id', 'cvFile']]
    df = featureExtraction.extract_skills_df (df)
    df = featureExtraction.resume_scoring(df,jobdesc)
    df = df.to_json(orient='records')
    df = json.loads(df)
    payload = {"scores" : df}

    response=requests.put(backend_endpoint+"/api/candidate/score-candidate",json=payload,headers=header)
    print(payload)
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


if __name__ == "__main__":
    uvicorn.run('main:app',reload=True)