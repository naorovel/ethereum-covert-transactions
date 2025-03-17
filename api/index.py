from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import uuid
from io import BytesIO
import pandas as pd
import numpy as np
import asyncio

app = FastAPI()
transactions_df = pd.read_csv("./src/data/transactions.csv")

@app.get("/api")
def hello_world():
    return {"message": "Hello World", "api": "Python"}

@app.get("/transaction")
def get_transaction(): 
    return {"transaction": "json object"}


@app.get("/get_transactions")
async def get_transactions(num_transactions: int): 
    
    display_transactions = transactions_df.head(num_transactions)
    # Put data processing here...
    
    result = display_transactions.to_csv()
    
    return result

