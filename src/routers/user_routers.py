import aiofiles
from fastapi import APIRouter, UploadFile

from recognition.controller import Controller
from models.model import RecognitionMethod, FileObject, QueryRespose



query_router = APIRouter(prefix='/query', tags=["User"])


@query_router.post('/')
async def query(files: list[UploadFile], recognition_method: RecognitionMethod) -> QueryRespose:
    
    """
    Processes uploaded files, filters out only those with the content type 'text/html',
    saves them to the disk, and returns a list of valid filenames.

    Args:
        files (list[UploadFile]): A list of uploaded files received via an HTTP request.

    Returns:
        Response: A JSON response containing a list of successfully saved filenames 
                  with the 'text/html' content type. The response is in the format 
                  {"filenames": ["filename1", "filename2", ...]}.
    """
    
    valid_files = []
    
    for file in files:
        if file.headers['content-type'] != 'text/html':
            continue
        
        async with aiofiles.open(f'static/{file.filename}', 'wb') as file_writter:
            content = await file.read()
            await file_writter.write(content)
        
            valid_files.append(FileObject(filename=file.filename, content=content.decode()))
        
    response = await Controller.resolve(valid_files, recognition_method)
            
    return response
