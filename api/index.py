from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gqlalchemy import Memgraph
import pandas as pd

app = FastAPI()
memgraph = Memgraph(host="localhost", port=7687)

transactions_df = pd.read_csv("./src/data/transactions.csv")

class QueryRequest(BaseModel):
    query: str
    
# Configure Memgraph connection
MEMGRAPH_HOST = "localhost"
MEMGRAPH_PORT = 7687

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

