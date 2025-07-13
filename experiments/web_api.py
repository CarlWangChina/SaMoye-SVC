from fastapi import FastAPI
from utils import getLogger

app = FastAPI()
logger = getLogger(__name__)


@app.post("/infer")
def infer():

    pass
