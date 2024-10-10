from fastapi import APIRouter, Response
import json


echo_router = APIRouter(prefix='/echo', tags=["Echo"])


@echo_router.post('/')
async def post_echo(message: str) -> Response:
    """
    Receives a message string and returns it in a JSON response.

    Args:
        message (str): The message string received from the HTTP request.

    Returns:
        Response: A JSON response containing the provided message in the format 
                  {"message": "your_message"}.
    """
    return Response(content=json.dumps({"message": message}), media_type='application/json')
