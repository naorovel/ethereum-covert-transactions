
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from json import loads, dumps

#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api.detection import run_detection_and_return_table

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


def update_graph(nodes, links, new_df):
    """ update node and link lsit with the new dataframe fragment
    """
    nodes = list(set(nodes) ^ set(new_nodes))
    return None
        
def get_nodes_links_from_df(transactions_df, start_idx, num_transactions):
    #display_transactions = transactions_df.head(num_transactions)
    display_transactions = transactions_df.iloc[start_idx:start_idx+num_transactions]
    transactions_df = display_transactions[["from_address","to_address", "value"]]
    # Renaming columns 
    unique_addr = transactions_df['from_address'].unique().tolist() + transactions_df['to_address'].unique().tolist()
    nodes = []
    for node in unique_addr: 
        nodes.append({'id': node})
    transactions_df["link"] = transactions_df.apply(lambda x: {'source': x["from_address"], 
                                                               'target': x["to_address"]}, axis=1)
    links = transactions_df["link"].values.tolist()
    return nodes, links



class QueryRequest(BaseModel):
    query: str

# @app.on_event("startup")
# async def startup():
#     app.state.txs_df = pd.read_csv("./src/data/transactions.csv") 
#     app.state.blocce_fund_txs_df = pd.read_csv("./api/fund_transactions.csv")
#     app.state.blocce_covert_txs_df = pd.read_csv("./api/blocce_transactions.csv") 
#     app.state.embedded_txs_df = pd.read_csv("./data/embedded_transactions.csv")
#     app.state.nodes = []
#     app.state.links = []
#     app.state.blocce_fund_txs_idx = 0
#     app.state.blocce_covert_txs_idx = 0


@app.get("/api")
def hello_world():
    return {"message": "Hello World", "api": "Python"}

@app.get("/transaction")
def get_transaction(): 
    return {"transaction": "json object"}

@app.get("/get_transactions")
async def get_transactions(num_transactions: int): 
    transactions_df = app.state.txs_df
    display_transactions = transactions_df.head(num_transactions)
    # Put data processing here...
    result = display_transactions.to_csv()
    return result


@app.get("/get_detected_transactions")
async def get_detected_transactions():
    df = run_detection_and_return_table()
    return loads(df.to_json(orient="records"))

@app.get("/get_table_transactions")
async def get_table_transactions(num_transactions: int): 
    transactions_df = app.state.txs_df
    display_transactions = transactions_df.head(num_transactions)
    # Put data processing here...
    table_transactions = display_transactions[["hash", "transaction_index", "from_address",
                                               "to_address", "value", "block_timestamp",
                                               "from_scam", "to_scam"]]
    result = table_transactions.to_json(orient="records")
    parsed = loads(result)
    return parsed

@app.get("/load_init_graph_transactions")
async def get_init_graph_transactions(num_transactions: int):
    print("/load_init_graph_transactions called")
    transactions_df = app.state.txs_df
    nodes, links = get_nodes_links_from_df(transactions_df, 0, num_transactions)
    # Save app state
    app.state.nodes.clear()
    app.state.links.clear()
    app.state.nodes += nodes
    app.state.links += links
    return {"message": f"Initial graph transactions loaded with {num_transactions} transactions."}

@app.get("/get_graph_transactions")
async def get_graph_transactions(num_transactions: int): 
    transactions_df = app.state.blocce_fund_txs_df 
    print(app.state.blocce_fund_txs_idx)
    nodes, links = get_nodes_links_from_df(transactions_df, app.state.blocce_fund_txs_idx, num_transactions)
    app.state.nodes += nodes
    app.state.links += links
    print("/get_graph_transactions called")
    print(len(app.state.nodes), len(app.state.links))
    app.state.blocce_fund_txs_idx += num_transactions
    return {'nodes': app.state.nodes, 'links': app.state.links}

@app.get("/fetch_graph_transactions")
async def fetch_graph_transactions(num_transactions=10): 
    transactions_df = app.state.blocce_fund_txs_df 
    print(app.state.blocce_fund_txs_idx)
    nodes, links = get_nodes_links_from_df(transactions_df, app.state.blocce_fund_txs_idx, num_transactions)
    app.state.nodes += nodes
    app.state.links += links
    print("/get_graph_transactions called")
    print(len(app.state.nodes), len(app.state.links))
    app.state.blocce_fund_txs_idx += num_transactions
    return {'nodes': app.state.nodes, 'links': app.state.links}

@app.get("/items")
def read_items():
    return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

@app.get("/inject_funding_transactions")
def inject_funding_transactions():
    
    return {"message": f"Injected funding transactions into graph"}

@app.get("/inject_covert_transactions")
def inject_covert_transactions():
    return True 

@app.get("/inject_normal_transactions")
def inject_normal_transactions():
    return True 

@app.get("/detect_and_remove_covert_transactions")
def detect_and_remove_covert_transactions():
    return True 





@app.get("/get_detected_transactions")
async def get_detected_transactions(num_transactions: int):
    detected_df = run_detection_and_return_table()
    display_df = detected_df.head(num_transactions)

    # Keep relevant columns
    table_data = display_df[["source", "target", "type", "covert_generated", "is_covert"]]

    # Convert to JSON and return
    result = table_data.to_json(orient="records")
    parsed = loads(result)
    return parsed





