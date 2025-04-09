from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from json import loads, dumps
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
        
table_df = None
graph_df = None
unique_addr = []
num_addr = 2000

def get_unique_addr(num_addr=10):
    global graph_df, unique_addr
    unique_addr = graph_df['source'].unique().tolist()
    print(len(unique_addr))
    # Remove duplicates 
    unique_addr = list(set(unique_addr))
    random.seed(4523)
    unique_addr = random.sample(unique_addr, num_addr)
    

def startup():
    global table_df, graph_df, num_addr
    table_df = pd.read_csv("./src/data/transactions.csv")
    graph_df = pd.read_csv("./src/data/graph.csv") 
    get_unique_addr(num_addr)
    
startup()       
        

## UTIL methods
def get_nodes_links_from_df(transactions_df):
    global unique_addr
    transactions_df = transactions_df[(transactions_df['source'].isin(unique_addr) | transactions_df['target'].isin(unique_addr))]
    #transactions_df = transactions_df[(transactions_df['source'].isin(unique_addr))]

    source_nodes = transactions_df['source'].tolist()
    target_nodes = transactions_df['target'].tolist()
    all_nodes = set(list(source_nodes + target_nodes))
    
    # Renaming columns 
    nodes = []
    for node in all_nodes: 
        nodes.append({'id': node})
    
    print(unique_addr)
    
    transactions_df["link"] = transactions_df.apply(lambda x: {'source': x["source"], 
                                                               'target': x["target"],
                                                               'type': x["type"],
                                                               'covert': x["covert"],
                                                               'covert_generated': x["covert_generated"],
                                                               }, axis=1)
    links = transactions_df["link"].values.tolist()
    return nodes, links

@app.get("/get_table_transactions")
async def get_table_transactions(): 
    global table_df, unique_addr

    transactions_df = table_df[table_df['from_address'].isin(unique_addr)]
        
    # Put data processing here...
    table_transactions = transactions_df[["hash", "transaction_index", "from_address",
                                               "to_address", "value", "block_timestamp"]]
    
    result = table_transactions.to_json(orient="records")
    
    parsed = loads(result)
    
    return parsed

@app.get("/get_graph_transactions")
async def get_graph_transactions(): 
    global graph_df
    nodes, links = get_nodes_links_from_df(graph_df)
    print(len(links))
    print(len(nodes))
    return {'nodes': nodes, 'links': links}



