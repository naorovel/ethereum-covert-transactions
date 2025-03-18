from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gqlalchemy import Memgraph
import pandas as pd
from json import loads, dumps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

transactions_df = pd.read_csv("./src/data/transactions.csv")

class QueryRequest(BaseModel):
    query: str

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

@app.get("/get_table_transactions")
async def get_table_transactions(num_transactions: int): 
    
    display_transactions = transactions_df.head(num_transactions)
    # Put data processing here...
    table_transactions = display_transactions[["hash", "transaction_index", "from_address",
                                               "to_address", "value", "block_timestamp",
                                               "from_scam", "to_scam"]]
    
    result = table_transactions.to_json(orient="records")
    parsed = loads(result)
    
    
    return parsed

@app.get("/get_graph_transactions")
async def get_graph_transactions(num_transactions: int): 
    
    display_transactions = transactions_df.head(num_transactions)
    # Put data processing here...
    
    result = display_transactions.to_csv()
    
    return result

@app.get("/items")
def read_items():
    return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]