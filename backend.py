from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import basic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class InterpreterType(BaseModel):
    type: str
    value: str|int

    def from_json(self):
        if self.type == "number":
            return basic.Number(value=self.value)
        elif self.type == "function":
            # TODO: this is probably wrong :(
            return basic.Function(name=self.name, body_node=None, arg_names=self.arg_names)
        else:
            raise NotImplementedError(f'missing implementation for {self.type}')

    def to_json(self):
        return {"type": type(self).__name__.lower(), "value": repr(self)}


class InterpretRequest(BaseModel):
    code: str
    symbols: Optional[Dict[str, InterpreterType]] = None


@app.post("/interpret")
def interpret(req: InterpretRequest)-> Dict:
    symbol_table = basic.SymbolTable.from_json(req.symbols)
    value, error, symbol_table = basic.run_ai("<stdin>", req.code, symbol_table=symbol_table)
    if error:
        return {"result": None, "symbol_table": None, "error": error.as_string()}

    return {
        "result": str(value) if value is not None else "",
        "symbols": symbol_table.to_json(),
        "error": None,
    }
