from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from tmatch.utils import gen_random_string

from .match import MatchData, Task, create_match_df

APP_RETURN_KEYS = ["sections", "embeddings", "ner"]

app = FastAPI()


def run_tasks(
    data: MatchData,
    task_id: str,
) -> JSONResponse:
    result = {"data": data.model_dump(), "task_id": task_id}
    try:
        data_dict = create_match_df(data, task_id=task_id).to_dict(orient="list")
        result.update(
            {
                k: v[0]
                for k, v in data_dict.items()
                if k in APP_RETURN_KEYS and len(v) < 2
            }
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {e}.\nDATA: {data.model_dump()}",
        )


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/talent_match/segment/")
def segment_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    data.tasks = [Task.SEGMENTATION]
    return run_tasks(data=data, task_id=task_id)


@app.post("/talent_match/embed/")
def job_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    data.tasks = [Task.EMBEDDINGS]
    return run_tasks(data=data, task_id=task_id)


@app.post("/talent_match/ner/")
def ner_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    data.tasks = [Task.NER]
    return run_tasks(data=data, task_id=task_id)


@app.post("/talent_match/all_tasks/")
def resume_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    data.tasks = [Task.SEGMENTATION, Task.EMBEDDINGS, Task.NER]
    return run_tasks(data=data, task_id=task_id)


@app.post("/talent_match/tasks/")
def tasks_endpoint(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    return run_tasks(data=data, task_id=task_id)


@app.post("/talent_match/bulk_tasks/")
def bulk_tasks_endpoint(
    data: MatchData, background_tasks: BackgroundTasks
) -> JSONResponse:
    if not data.segs_folder or not data.ems_folder or not data.ner_folder:
        raise HTTPException(
            status_code=400,
            detail=f"All of segs_folder, ems_folder, and ner_folder must be provided.\nDATA: {data.model_dump()}",
        )
    data.text = ""
    task_id = gen_random_string()
    background_tasks.add_task(run_tasks, data=data, task_id=task_id)
    return JSONResponse(content={"data": data.model_dump(), "task_id": task_id})


"""
The fields that the endpoints accept are:

data_path: The path to the file or directory containing the data to be processed.
text: The text to be processed. If it is provided, it will be used instead of the data_path.
text_file_name: The name of the file to write the text to. It is used when the text is provided. It is a random string by default.
extension: The extension of the file to write the text to. It is used when the text is provided.
           If it is one of '.pdf', '.doc', or '.docx', we will assume that the text is base64 encoded.
           It will then be decoded and written to a file with the given extension.
           If it's '.txt', the text will be written to a file with the given extension. It is '.txt' by default.
tenantid: The tenant id of the user.
tasks: The list of tasks to perform. The possible tasks are: 'SEGMENTATION', 'EMBEDDINGS', and 'NER'
segs_folder: The folder to write the segmentation results to. 
             If it is not provided, the segmentation results will not be written to a file. 
             It must be provided when using the resumes/ endpoint.
ems_folder: The folder to write the embeddings to.
            If it is not provided, the embeddings will not be written to a file.
            It must be provided when using the resumes/ endpoint.
ner_folder: The folder to write the named entity recognition results to.
            If it is not provided, the named entity recognition results will not be written to a file.
            It must be provided when using the resumes/ endpoint.
meta_data: The metadata to be added to the text segments.

The endpoints are:

/talent_match/segment/: It sets tasks to just ['SEGMENTATION'] so it extracts the segmentation of data_path or text and returns the segmentation. It writes the segmentation too if segs_folder is provided.
/talent_match/embed/: It sets tasks to just ['EMBEDDINGS'] so it extracts the embeddings of data_path or text and returns the embeddings. It writes the embeddings too if ems_folder is provided.
/talent_match/ner/: It sets tasks to just ['NER'] so it extracts the named entity recognition results of data_path or text and returns the results. It writes the named entity recognition results too if ner_folder is provided.
/talent_match/all_tasks/: It sets tasks to ['SEGMENTATION', 'EMBEDDINGS', 'NER'] so it extracts the segmentation, embeddings, and named entity recognition results of data_path or text and returns the results. It writes the results too if the corresponding folders are provided.
/talent_match/tasks/: The possible tasks are: 'SEGMENTATION', 'EMBEDDINGS', and 'NER'. It extracts the specified tasks of data_path or text and returns the results. It writes the results too when the corresponding folders are provided.
/talent_match/bulk_tasks/: The possible tasks are: 'SEGMENTATION', 'EMBEDDINGS', and 'NER'. It extracts the specified tasks of data_path or text and returns the results. It writes the results too when the corresponding folders are provided.

All endpoints return a JSON response with the following fields:

data: The data that was sent to the endpoint.
task_id: The id of the task that was created. It is a random string.

The segment/ endpoint also returns the segmentation.
The embed/ endpoint also returns the embeddings.
The ner/ endpoint also returns the named entity recognition results.
The all_tasks/ and tasks/ endpoints also return the results of the tasks that were performed.
The bulk_tasks/ endpoint does not return the results of the tasks that were performed. It only returns the data and the task_id.
"""
