import json
from pathlib import Path

from termcolor import colored

from .redis_kv_store import KeyValueStore

PROGRESS_KEY = "pdf_text_progress"
STATUS_KEY = "pdf_text_status"
ERROR_KEY = "pdf_text_error_files"
COUNT_NAME = "doc_count"
INC = 1


def prepare_task(id_key: str, task_data: dict, kv_store: KeyValueStore):
    task_id = task_data.get(id_key, None)
    if task_id is None:
        raise Exception("No task_id found in task_data")
    try:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    except Exception as e:
        raise Exception(e)
    data = task_data.get("data", None)
    if data is None:
        raise Exception(f"No data found for task_id {task_id}.")
    task["status"] = "TASK_STATUS_IN_PROGRESS"
    # kv_store.insert(task_id, task)
    return task_id, task, data


def finish_task(task_id: str, kv_store: KeyValueStore, results):
    print(colored(f"\n\n*******RESULTS = {results}\n\n", "green"))
    try:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    except Exception as e:
        raise Exception(f"FINISH_TASK: GET TASK STATUS FAILED WITH STATUS {e}")
    try:
        task["status"] = "TASK_STATUS_FINISHED"
        task["results"] = json.dumps(results)
        print(colored(f"\n\n*******TASK_ID = {task_id}*********\n\n", "cyan"))
        kv_store.insert(task_id, task)
    except Exception as e:
        print(
            colored(f"\n\nUPDATING FINAL TASK STATUS FAILED WITH ERROR {e}\n\n", "red")
        )
    print(colored("\n\n*******AFTER INSERTING*******\n\n", "cyan"))


def failed_task(task_id: str, kv_store: KeyValueStore, e: Exception):
    print(colored(f"\n\nACTION FAILED WITH ERROR {e}\n\n", "red"))
    try:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    except Exception as e:
        print(colored(f"SETTING TASK ERROR STATUS FAILED WITH ERROR: {e}", "red"))
        return
    task["status"] = "TASK_STATUS_ERROR"
    task["error"] = f"action failed with error {e}"
    kv_store.insert(task_id, task)


def get_task_from_kv_store(task_id: str, kv_store: KeyValueStore) -> dict:
    task = kv_store.get(task_id)
    if task is None:
        raise Exception(f"No task entry found for task_id {task_id}.")
    if not isinstance(task, dict):
        raise Exception(f"Wrong type for task with task_id {task_id}.")
    if len(task) == 0:
        raise Exception(f"Empty dict for task_id {task_id}.")
    return task


def init_task_progress(
    task_id: str,
    kv_store: KeyValueStore,
    df_len: int,
    task_key: str = PROGRESS_KEY,
) -> None:
    if task_id is not None and kv_store is not None:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
        task[task_key] = f"0/{df_len}"
        kv_store.insert(task_id, task)


def update_progress(task: dict, task_key: str = PROGRESS_KEY, inc: int = INC) -> dict:
    prog = task.get(task_key, "0/total")
    curr, total = prog.split("/")
    task[task_key] = f"{int(curr) + inc}/{total}"
    return task


def get_task_progress(
    task_id: str, kv_store: KeyValueStore, task_key: str = PROGRESS_KEY
) -> tuple[int, int]:
    task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    prog = task.get(task_key, "0/total")
    curr, total = prog.split("/")
    return int(curr), int(total)


def update_task_progress(
    task_id: str,
    kv_store: KeyValueStore,
    task_key: str = PROGRESS_KEY,
    inc: int = INC,
    total: int | str = "total",
) -> None:
    if task_id is not None and kv_store is not None:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
        prog = task.get(task_key, f"0/{total}")
        curr, total = prog.split("/")
        task[task_key] = f"{int(curr) + inc}/{total}"
        kv_store.insert(task_id, task)


def update_status(
    task_id: str,
    kv_store: KeyValueStore,
    status: str,
    task_key: str = STATUS_KEY,
) -> None:
    if task_id is not None and kv_store is not None:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
        task[task_key] = status
        kv_store.insert(task_id, task)


def update_task_error_files(
    task_id: str,
    kv_store: KeyValueStore,
    file: str,
    task_key: str = ERROR_KEY,
) -> None:
    if task_id is not None and kv_store is not None:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
        error_files = task.get(task_key, "")
        if error_files != "":
            error_files += ", "
        error_files += str(file)
        task[task_key] = error_files
        kv_store.insert(task_id, task)


def kv_store_update_error_files(docx: str | Path, task_data: dict):
    task_id = task_data.get("task_id", None)
    kv_store = task_data.get("kv_store", None)
    if task_id is None:
        return
    if kv_store is None:
        return None
    try:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    except Exception as e:
        print(colored(f"Error {e} updating kv store for task_id: {task_id}", "red"))
        return
    error_files = task.get("error_files", "")
    if error_files != "":
        error_files += ", "
    error_files += str(docx)
    task["error_files"] = error_files
    kv_store.insert(task_id, task)


def kv_store_update_count(task_data: dict, name: str = COUNT_NAME) -> None:
    task_id = task_data.get("task_id", None)
    kv_store = task_data.get("kv_store", None)
    if task_id is None:
        return
    if kv_store is None:
        return None
    try:
        task = get_task_from_kv_store(task_id=task_id, kv_store=kv_store)
    except Exception as e:
        print(colored(f"Error {e} updating kv store for task_id: {task_id}", "red"))
        return
    count = task.get(name, 0)
    task[name] = int(count) + 1
    kv_store.insert(task_id, task)
