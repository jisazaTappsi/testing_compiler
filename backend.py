from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import basic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class InterpretRequest(BaseModel):
    code: str


@app.post("/interpret")
def interpret(req: InterpretRequest):
    value, error = basic.run_ai("<stdin>", req.code)
    if error:
        return {"result": None, "error": error.as_string()}
    return {"result": str(value) if value is not None else "", "error": None}
