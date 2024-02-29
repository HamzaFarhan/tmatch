import base64
import os
import re
import shutil
import subprocess
import uuid
from inspect import cleandoc
from itertools import chain
from pathlib import Path
from textwrap import dedent
from typing import Iterator

import demoji
from bs4 import BeautifulSoup
from fastcore.basics import flatten
from unstructured.cleaners.core import (
    bytes_string_to_string,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    group_bullet_paragraph,
    replace_mime_encodings,
    replace_unicode_quotes,
)

RANDOM_STRING_LENGTH = 16


def b64_to_file(b64, path: Path | str) -> str:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path


def gen_random_string(length: int = RANDOM_STRING_LENGTH) -> str:
    return str(uuid.uuid4()).replace("-", "")[:length]


def flatten_list(my_list: list) -> list:
    "Flatten a list of lists."
    l2 = []
    for x in my_list:
        if isinstance(x, list):
            l2 += flatten_list(x)
        else:
            l2.append(x)
    return l2


def deindent(text: str) -> str:
    return dedent(cleandoc(text))


def resolve_data_path(data_path: str | list[str]) -> Iterator[Path]:
    """
    Resolve a list of data paths to a list of file paths.

    Parameters
    ----------
    data_path : str or List[str]
        The path or list of paths to the data.

    Returns
    -------
    Iterator[Path]
        An iterator over the resolved file paths.

    Raises
    ------
    Exception
        If a path in the list does not exist.

    """
    if not isinstance(data_path, list):
        data_path = [data_path]
    data_path = flatten(data_path)
    paths = []
    for dp in data_path:
        if isinstance(dp, (Path, str)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                paths.append(dp.iterdir())
            else:
                paths.append([dp])
    return chain(*paths)


def has_html_tags(text: str) -> bool:
    if "<" in text and ">" in text and "</" in text:
        return True
    else:
        return False


def remove_html_tags(text: str) -> str:
    if not has_html_tags(text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="")


def clean_text(
    text: str,
    remove_html: bool = True,
    include_emojis: bool = False,
    group: bool = False,
) -> str:
    text = re.sub(r"[\n+]", "\n", text)
    text = re.sub(r"[\t+]", " ", text)
    text = re.sub(r"[. .]", " ", text)
    text = re.sub(r"([ ]{2,})", " ", text)
    # print(text)
    try:
        text = bytes_string_to_string(text)
        # print(text)
    except Exception:
        pass
    try:
        text = clean_non_ascii_chars(text)
        # print(text)
    except Exception:
        pass
    try:
        text = replace_unicode_quotes(text)
        # print(text)
    except Exception:
        pass
    try:
        text = replace_mime_encodings(text)
        # print(text)
    except Exception:
        pass
    if group:
        try:
            text = "\n".join(group_bullet_paragraph(text))
            # print(text)
        except Exception:
            pass
        try:
            text = group_broken_paragraphs(text)
            # print(text)
        except Exception:
            pass
    if not include_emojis:
        text = demoji.replace(text, "")
    if remove_html:
        try:
            text = remove_html_tags(text)
        except Exception:
            pass
    return text


def setify(o):
    return o if isinstance(o, set) else set(list(o))


def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path: str | Path,
    extensions: list[str] = None,
    recurse: bool = True,
    folders: list[str] = None,
    followlinks: bool = True,
    make_str: bool = False,
) -> list:
    if folders is None:
        folders = list([])
    path = Path(path)
    if extensions is not None:
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)
    if make_str:
        res = [str(o) for o in res]
    return list(res)


def json_file(path: str | Path, folder: str | Path) -> str:
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    return str((folder / Path(path).name).with_suffix(".json"))


def is_bucket(p: str | Path) -> bool:
    return str(p).startswith("gs://")


def gsutil_bucket(bucket: str | Path) -> str:
    if not str(bucket).startswith("gs://"):
        bucket = "gs://" + str(bucket)
    if bucket[-1] != "/":
        bucket += "/"
    return bucket


def gsutil_src(local_folder: str | Path) -> str:
    if Path(local_folder).suffix != "":
        return str(local_folder)
    local_folder = str(local_folder)
    if local_folder[-1] != "/":
        local_folder += "/"
    local_folder += "*"
    return local_folder


def bucket_move(local_folder: str | Path, bucket: str) -> None:
    gu = shutil.which("gsutil")
    bucket = gsutil_bucket(bucket)
    if Path(local_folder := gsutil_src(local_folder)).suffix == "":
        files = get_files(local_folder, make_str=True)
        if len(files) == 1:
            local_folder = files[0]
    subprocess.run([gu, "-m", "mv", local_folder, bucket])


def bucket_up(local_folder: str | Path, bucket: str, only_new: bool = True) -> None:
    gu = shutil.which("gsutil")
    bucket = gsutil_bucket(bucket)
    cmd = [gu, "-m", "cp"]
    if only_new:
        cmd.append("-n")
    if Path(local_folder := gsutil_src(local_folder)).suffix == "":
        files = get_files(local_folder, make_str=True)
        if len(files) == 1:
            local_folder = files[0]
        else:
            cmd.append("-r")
    subprocess.run(cmd + [local_folder, bucket])


def bucket_dl(bucket: str, local_folder: str | Path, only_new: bool = True) -> None:
    gu = shutil.which("gsutil")
    bucket = gsutil_bucket(bucket)
    bucket = gsutil_src(bucket)
    os.makedirs(local_folder, exist_ok=True)
    cmd = [gu, "-m", "cp", "-r"]
    if only_new:
        cmd.append("-n")
    subprocess.run(cmd + [bucket, local_folder])


def get_local_path(remote_folder: str | Path, local_folder: str | Path) -> Path:
    if not remote_folder or not local_folder:
        return ""
    if is_bucket(remote_folder):
        if Path(remote_folder).suffix != "":
            local_path = Path(local_folder)
        else:
            local_path = Path(local_folder) / Path(remote_folder).name
    else:
        local_path = Path(remote_folder)
    if not local_path.exists():
        local_path.mkdir(parents=True, exist_ok=True)
    return local_path


def count_words(text: str) -> int:
    return len(text.split())


def count_lines(text: str) -> int:
    return len(text.split("\n"))


def token_count_to_word_count(token_count) -> int:
    return max(int(token_count * 0.75), 1)


def token_count_to_line_count(token_count) -> int:
    return max(int(token_count * 0.066), 1)


def word_count_to_token_count(word_count) -> int:
    return max(int(word_count / 0.75), 1)


def count_tokens(text: str) -> int:
    return word_count_to_token_count(len(text.split()))
