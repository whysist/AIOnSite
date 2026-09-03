from fastapi import FastAPI


app = FastAPI(title="AIOnSite")


@app.get("/health")
async def health():
    return {"status": "ok"}
