from fastapi import APIRouter, Response
import json


echo_router = APIRouter(prefix='/echo', tags=["Echo"])


@echo_router.post('/')
async def post_echo(message: str) -> Response:
    """
    It'll just return what you send. Just to make sure it's working.
    """
    return Response(content=json.dumps({"message": message}), media_type='application/json')
