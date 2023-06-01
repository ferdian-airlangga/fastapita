from fastapi import FastAPI, UploadFile, File
import uvicorn
import featureExtraction
import pandas as pd
import requests
import PyPDF2
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/resume_scoring")
async def resume_scoring(positionId: str):    
    
    backend_endpoint = "https://5815-182-253-194-86.ngrok-free.app"

    payload = {
        "email" : "warren@gmail.com",
        "password" : "000000"
    }


    response=requests.post(backend_endpoint+"/api/auth/login",json=payload)
    json_data = response.json()
    token=json_data['token']

    payload = {
        "id" : positionId
    }
    
    header = {'Authorization': token}
    response=requests.get(backend_endpoint+"/api/position/get-one-position",json=payload,headers=header)
    json_data = response.json()
    jobdesc = json_data["description"]+" "+json_data["qualification"]

    payload = {
        "positionId" : positionId
    }

    response=requests.get(backend_endpoint+"/api/candidate/get-candidate",json=payload,headers=header)
    json_data = response.json()
    
    df = pd.DataFrame(json_data)
    df = df[['_id', 'cvFile']]
    df = featureExtraction.extract_skills_df (df)
    result = featureExtraction.resume_scoring(df,jobdesc)

    payload = {
        "scores" : result
    }

    for i in payload['scores'] :
        i['id']=i.pop('_id')

    header = {'Authorization': token}
    response=requests.put(backend_endpoint+"/api/candidate/score-candidate",json=payload,headers=header)

    return response
    
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