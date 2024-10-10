from fastapi import FastAPI

from routers.default_routers import echo_router
from routers.user_routers import query_router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(echo_router)
app.include_router(query_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == '__main__':
    import uvicorn  

    uvicorn.run(app="main:app", host='0.0.0.0', port=2000, reload=True, workers=2)