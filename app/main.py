from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from tmatch.utils import gen_random_string

from .match import MatchData, Task, create_match_df

TASKS = [Task.SEGMENTATION, Task.EMBEDDINGS, Task.NER]
APP_RETURN_KEYS = ["sections", "embeddings", "ner"]

app = FastAPI()


def action(
    data: MatchData,
    task_id: str,
    tasks: list[Task] = TASKS,
) -> JSONResponse:
    result = {"data": data.model_dump(), "task_id": task_id}
    try:
        data_dict = create_match_df(data, task_id=task_id, tasks=tasks).to_dict(
            orient="list"
        )
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


@app.post("/talent_match/jobs")
def job_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    tasks = [Task.EMBEDDINGS]
    return action(data=data, task_id=task_id, tasks=tasks)


@app.post("/talent_match/search_text")
def search_text_action(data: MatchData) -> JSONResponse:
    return job_action(data)


@app.post("/talent_match/resume/")
def resume_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    return action(data=data, task_id=task_id)


@app.post("/talent_match/ner/")
def ner_action(data: MatchData) -> JSONResponse:
    task_id = gen_random_string()
    tasks = [Task.NER]
    return action(data=data, task_id=task_id, tasks=tasks)


@app.post("/talent_match/resumes/")
def resumes_action(data: MatchData, background_tasks: BackgroundTasks) -> JSONResponse:
    if not data.segs_folder or not data.ems_folder or not data.ner_folder:
        raise HTTPException(
            status_code=400,
            detail=f"All of segs_folder, ems_folder, and ner_folder must be provided.\nDATA: {data.model_dump()}",
        )
    data.text = ""
    task_id = gen_random_string()
    background_tasks.add_task(action, data=data, task_id=task_id)
    return JSONResponse(content={"data": data.model_dump(), "task_id": task_id})
