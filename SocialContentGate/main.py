from fastapi import FastAPI
import uvicorn

from config import *

from vkapi import *
from tgapi import *
from instapi import *
from twogisapi import *

app = FastAPI()

@app.get("/")
async def root():
    return {"Success": True}


@app.get("/healthcheck")
async def healthcheck():
    return {"success": True}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
