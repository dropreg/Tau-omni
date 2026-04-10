from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import threading

app = FastAPI()

# -------- 全局共享状态 --------
store: Dict[str, Dict[str, object]] = {}
store_lock = threading.Lock()

# -------- 请求 / 响应模型 --------
class SyncRequest(BaseModel):
    id: str
    golden_idx: str

class SyncResponse(BaseModel):
    id: str
    access_index: int
    golden_idx: str

class DeleteResponse(BaseModel):
    id: str
    deleted: bool

# -------- 接口实现 --------
@app.post("/sync", response_model=SyncResponse)
def sync(req: SyncRequest):
    with store_lock:
        if req.id not in store:
            # 第一次访问
            store[req.id] = {
                "access_count": 0
            }
            access_index = 0
            print(f"[debug] {req.id} not in store!!!😭{access_index} in {len(store)}")
        else:
            
            store[req.id]["access_count"] += 1
            access_index = min(store[req.id]["access_count"], 3)
            print(f"[debug] {req.id} in store!!!😄{access_index} in {len(store)}")

    return SyncResponse(
        id=req.id,
        access_index=access_index,
        golden_idx=req.golden_idx
    )

@app.delete("/delete/{id}", response_model=DeleteResponse)
def delete_by_id(id: str):
    with store_lock:
        if id not in store:
            return DeleteResponse(id=id, deleted=False)

        del store[id]
        print(f"[debug] delete {id} in store!!!😄 {len(store)}")
        return DeleteResponse(id=id, deleted=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)