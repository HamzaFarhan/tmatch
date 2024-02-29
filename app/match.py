import os
import shutil
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from termcolor import colored

from tmatch import redis_kv_store
from tmatch.extract import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    create_path_df,
    extract_resume_text,
)
from tmatch.gen import (
    gen_embeddings,
    gen_ner_sections,
    gen_sections,
    write_df_ems,
    write_df_sections,
)
from tmatch.task_utils import (
    get_task_from_kv_store,
    init_task_progress,
    update_status,
    update_task_error_files,
    update_task_progress,
)
from tmatch.utils import (
    b64_to_file,
    bucket_dl,
    gen_random_string,
    get_local_path,
    is_bucket,
)

TASK_FOLDER = Path("/tmp")
EXTENSIONS = [".pdf", ".doc", ".docx"]


def add_slash(s: str) -> str:
    if not s:
        return s
    return s if s.endswith("/") else s + "/"


class Task(str, Enum):
    EXTRACTION = "EXTRACTION"
    SEGMENTATION = "SEGMENTATION"
    EMBEDDINGS = "EMBEDDINGS"
    NER = "NER"


class MatchData(BaseModel):
    data_path: str = ""
    text: str = "My name is Mike Deen from LA and I am a Machine Learning Dev good at python and pytorch working at Netflix since 2019 and I have a BSCS from MIT. I previously worked at Google."
    text_file_name: str = Field(default_factory=gen_random_string)
    extension: str = ".txt"
    tenantid: str = Field(
        title="The tenant id of the user.",
        example="ten_1234",
        default="ten_1234",
    )
    tasks: list[Task] = [Task.SEGMENTATION, Task.EMBEDDINGS, Task.NER]
    segs_folder: str = Field(
        title="The folder to write the segmentation results to.",
        description="It must be a directory.",
        default="",
    )
    ems_folder: str = Field(
        title="The folder to write the embeddings to.",
        description="It must be a directory.",
        default="",
    )
    ner_folder: str = Field(
        title="The folder to write the named entity recognition results to.",
        description="It must be a directory.",
        default="",
    )
    meta_data: dict = {}
    # task_folder: str = str(TASK_FOLDER)
    # task_id: str = Field(default_factory=gen_random_string)

    @model_validator(mode="after")
    def process_data(self) -> "MatchData":
        # self.task_folder = Path(self.task_folder) / self.task_id
        # os.makedirs(self.task_folder, exist_ok=True)
        self.segs_folder = add_slash(self.segs_folder)
        self.ems_folder = add_slash(self.ems_folder)
        self.ner_folder = add_slash(self.ner_folder)
        if not self.extension.startswith("."):
            self.extension = "." + self.extension
        self.text_file_name = str(Path(self.text_file_name).with_suffix(self.extension))
        # if self.text:
        #     text_path = str(self.task_folder / self.text_file_name)
        #     if self.extension in EXTENSIONS:
        #         self.data_path = str(b64_to_file(self.text, path=text_path))
        #     else:
        #         self.data_path = text_path
        #         with open(self.data_path, "w") as f:
        #             f.write(self.text)
        # self.task_folder = str(self.task_folder)
        return self


def apply_fns_to_df_row(
    row: pd.Series,
    fns: Callable | list[Callable],
    # apply_cols: str | list[str],
    # new_cols: str | list[str],
    df_len: int = 0,
    task: Task = Task.EXTRACTION,
    task_id=None,
    kv_store=None,
    redis_host=redis_kv_store.REDIS_HOST,
    redis_port=redis_kv_store.REDIS_PORT,
) -> pd.Series:
    file_path = str(row["path"])
    if redis_host and redis_port:
        kv_store = redis_kv_store.KeyValueStore(
            redis_host=redis_host, redis_port=redis_port
        )
    update_status(
        task_id=task_id,
        kv_store=kv_store,
        status=task.value,
        task_key="DATA_STATUS",
    )
    try:
        for fn in [fns] if not isinstance(fns, list) else fns:
            row = fn(row)
        update_task_progress(
            task_id=task_id,
            kv_store=kv_store,
            task_key=f"{task.value}_PROGRESS",
            total=df_len,
        )
    except Exception as e:
        print(colored(f"ERROR with {file_path}: {e}", "red"))
        update_task_error_files(
            task_id=task_id,
            kv_store=kv_store,
            file=file_path,
            task_key=f"{task.value}_ERROR_FILES",
        )
        # row[new_col] = None
    print(
        colored(f"\n\nTASK: {get_task_from_kv_store(task_id, kv_store)}\n\n", "green")
    )
    return row


def row_fn(
    row: pd.Series, fn: Callable, apply_col: str = "", new_col: str = ""
) -> pd.Series:
    if apply_col:
        res = fn(row[apply_col])
    else:
        res = fn(row)
    if new_col:
        row[new_col] = res
    return row


def create_match_df(
    data: MatchData,
    task_id: str = "",
    chunk_size: int | None = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    redis_host: str = redis_kv_store.REDIS_HOST,
    redis_port: int = redis_kv_store.REDIS_PORT,
) -> pd.DataFrame:
    print(f"\n\nMATCH DATA: {data.model_dump()}\n\n")

    if not data.data_path and not data.text:
        return pd.DataFrame()

    task_id = task_id or gen_random_string()
    os.makedirs(task_folder := Path(TASK_FOLDER) / task_id, exist_ok=True)

    kv_store = None
    if redis_host and redis_port:
        kv_store = redis_kv_store.KeyValueStore(
            redis_host=redis_host, redis_port=redis_port
        )
        kv_store.insert(task_id, {"data_status".upper(): "STARTED"})
        task = get_task_from_kv_store(task_id, kv_store)
        task = {k: v for k, v in task.items() if "progress" not in k.lower()}
        print(colored(f"KVSTORE TASK: {task}", "cyan"))
        kv_store.insert(task_id, task)

    if data.text:
        text_path = str(task_folder / data.text_file_name)
        if data.extension in EXTENSIONS:
            data.data_path = str(b64_to_file(data.text, path=text_path))
        else:
            data.data_path = text_path
            with open(data.data_path, "w") as f:
                f.write(data.text)

    local_data_path = get_local_path(
        remote_folder=data.data_path, local_folder=task_folder
    )
    local_segs_folder = get_local_path(
        remote_folder=data.segs_folder, local_folder=task_folder
    )
    local_ems_folder = get_local_path(
        remote_folder=data.ems_folder, local_folder=task_folder
    )
    local_ner_folder = get_local_path(
        remote_folder=data.ner_folder, local_folder=task_folder
    )

    if is_bucket(data.data_path):
        bucket_dl(bucket=data.data_path, local_folder=local_data_path)

    df = create_path_df(
        data_path=local_data_path,
        extra_cols={
            "segs_folder": local_segs_folder,
            "ems_folder": local_ems_folder,
            "ner_folder": local_ner_folder,
        },
    )
    df["tenantid"] = data.tenantid

    row_fns = {
        Task.EXTRACTION: lambda row: row_fn(
            row,
            partial(
                extract_resume_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            apply_col="path",
            new_col="text",
        ),
        Task.SEGMENTATION: [
            lambda row: row_fn(
                row=row,
                fn=partial(gen_sections, sep="", existing_sections=data.meta_data),
                apply_col="text",
                new_col="sections",
            ),
            lambda row: row_fn(
                row=row,
                fn=partial(
                    write_df_sections,
                    sections_col="sections",
                    sections_folder=local_segs_folder,
                    remote_sections_folder=data.segs_folder,
                ),
            ),
        ],
        Task.EMBEDDINGS: [
            lambda row: row_fn(
                row=row,
                fn=gen_embeddings,
                apply_col="sections" if "sections" in row else "text",
                new_col="embeddings",
            ),
            lambda row: row_fn(
                row=row,
                fn=partial(
                    write_df_ems,
                    ems_folder=local_ems_folder,
                    remote_ems_folder=data.ems_folder,
                ),
            ),
        ],
        Task.NER: [
            lambda row: row_fn(
                row=row,
                fn=gen_ner_sections,
                apply_col="text",
                new_col="ner",
            ),
            lambda row: row_fn(
                row=row,
                fn=partial(
                    write_df_sections,
                    sections_col="ner",
                    sections_folder=local_ner_folder,
                    remote_sections_folder=data.ner_folder,
                ),
            ),
        ],
    }
    apply_args = {
        "df_len": len(df),
        "task_id": task_id,
        "kv_store": kv_store,
    }
    if data.text:
        df["text"] = [[data.text]]
    else:
        init_task_progress(**apply_args, task_key=f"{Task.EXTRACTION.value}_PROGRESS")
        df = df.apply(
            lambda row: apply_fns_to_df_row(
                row=row,
                fns=row_fns[Task.EXTRACTION],
                task=Task.EXTRACTION,
                **apply_args,
            ),
            axis=1,
        )

    if data.segs_folder or Task.SEGMENTATION in data.tasks:
        init_task_progress(**apply_args, task_key=f"{Task.SEGMENTATION.value}_PROGRESS")
        df = df.apply(
            lambda row: apply_fns_to_df_row(
                row=row,
                fns=row_fns[Task.SEGMENTATION],
                task=Task.SEGMENTATION,
                **apply_args,
            ),
            axis=1,
        )
    if data.ems_folder or Task.EMBEDDINGS in data.tasks:
        init_task_progress(**apply_args, task_key=f"{Task.EMBEDDINGS.value}_PROGRESS")
        df = df.apply(
            lambda row: apply_fns_to_df_row(
                row=row,
                fns=row_fns[Task.EMBEDDINGS],
                task=Task.EMBEDDINGS,
                **apply_args,
            ),
            axis=1,
        )
    if data.ner_folder or Task.NER in data.tasks:
        init_task_progress(**apply_args, task_key=f"{Task.NER.value}_PROGRESS")
        df = df.apply(
            lambda row: apply_fns_to_df_row(
                row=row,
                fns=row_fns[Task.NER],
                task=Task.NER,
                **apply_args,
            ),
            axis=1,
        )
    if kv_store:
        kv_store.insert(task_id, {"data_status".upper(): "COMPLETED"})
    shutil.rmtree(task_folder)
    return df
