import json
import re
from pathlib import Path
from typing import Callable

import google.generativeai as genai
import numpy as np
import pandas as pd
from termcolor import colored

from .extract import MAX_CHUNK_SIZE
from .llms import ModelName, ask_llm, system_message, user_message
from .utils import bucket_up, deindent, flatten_list, is_bucket

SECTIONS = ["Education", "Work Experience", "Skills", "Summary"]
NER_SECTIONS = [
    "Companies worked at (Just the names. No extra info)",
    "Job Titles (Just the titles. No extra info)",
    "Skills",
    "Certifications (Just the names. No extra info)",
    "Educational Institutions (Just the names. No extra info)",
    "Highest Degree (Just the degree. No extra info)",
    "Full Name (Just the name. No extra info)",
    "Email",
    "Phone Number",
    "Location",
]
SEP = "<br>"
EMS_MODEL = "multi-qa-mpnet-base-cos-v1"
GOOGLE_EMS_MODEL = "models/embedding-001"


def split_comma_separated_list(data: str) -> list[str]:
    data = data.replace("\n", ",")
    return [r.strip() for r in data.split(",") if r.strip()]


def get_text_between_tags(text: str, tag1: str, tag2: str) -> str:
    tag1 = str(tag1)
    tag2 = str(tag2)
    if tag1 in text and tag2 in text:
        return deindent(text.split(tag1)[1].split(tag2)[0].strip())
    return ""


def gen_tagged_response(
    data: str, prompt: str, tag: str = "<response>", model: str = ModelName.GEMINI
) -> str:
    tag1 = tag
    if not tag.startswith("<"):
        tag1 = f"<{tag1}>"
    tag2 = tag1.replace("<", "</")
    messages = [
        system_message(
            f"You are an expert data extractor. Enclose you response in {tag1} {tag2}. If there is no response, return {tag1}{tag2}."
        ),
        user_message(prompt),
        user_message(data),
    ]
    try:
        res = ask_llm(messages=messages, model=model)[-1]["content"]
        print(colored(res, "green"))
    except Exception as e:
        print(colored(e, "red"))
        return ""
    return get_text_between_tags(res, tag1=tag1, tag2=tag2)


def split_name(name: str) -> tuple[str, str]:
    split_name = [n.strip() for n in name.split()]
    first_name_parts = [split_name[0]]

    for n in split_name[1:-1]:
        if len(n) <= 2 and n.islower():
            first_name_parts.append(n)

    first_name = " ".join(first_name_parts)

    last_name = (
        split_name[-1]
        if len(split_name) > 1 and not first_name.endswith(split_name[-1])
        else ""
    )

    return first_name.strip(), last_name.strip()


def sections_prompt_fn(
    sections: list[str] = SECTIONS,
    sep: str = SEP,
) -> str:
    task = deindent(
        """
        As an expert recruiter, organize this resume text into detailed sections without excluding any content.
        Missing any content would be a disservice to the candidate.
        The sections should be:
        """
    )
    sections_str = "\n".join([f"    {i+1}. {s}" for i, s in enumerate(sections)])
    tags_prompt = "Enclose each section into tags like this:"
    section_tags_str = "\n".join(
        [f"    <{s}>\n    ...\n    </{s}>\n" for s in sections]
    )
    format_prompt = deindent(
        """
        Don't format the text as markdown or anything else. Just use plain text. Remember to add spaces between words.
        I'll use your tags to organize the text into sections. So make sure to use the tags correctly.
        It should be like: <section1> ... </section1> <section2> ... </section2> ...
        NOT like: <section1> ... <section2> ... </section2> ... </section1> ...
        """
    )
    if sep:
        format_prompt = deindent(
            f"{format_prompt}\nIf a section has multiple entries, add a {sep} tag between them."
        )
    personal_prompt = deindent(
        """
        If 'Personal Information' is one of the sections, it's very important to get it right.
        """
    )
    exp_prompt = deindent(
        """
        If 'Work Experience' is one of the sections, it's very important to get it right.
        Make sure to include the job details as well. Not just the job title and company.
        But don't make up any information. If the details are not there, just leave it empty.
        """
    )
    edu_prompt = deindent(
        """
        If 'Education' is one of the sections, it's very important to get it right.
        Make sure to include the degree details as well. Not just the school name.
        But don't make up any information. If the details are not there, just leave it empty.
        """
    )
    empty_prompt = "If a section is missing, just return <section> </section> with a space in between."
    return deindent(
        f"""
{task}
{sections_str}

{tags_prompt}
{section_tags_str}
{personal_prompt}
{exp_prompt}
{edu_prompt}

{format_prompt}
{empty_prompt}
"""
    )


def gen_sections_(
    text: str,
    sections: list[str] = SECTIONS,
    model: str = ModelName.GEMINI,
    sep: str = SEP,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> dict[str, str]:
    messages = [
        user_message(sections_prompt_fn(sections=sections, sep=sep)),
        user_message(deindent(f"RESUME TEXT:\n{text[:max_chunk_size]}")),
    ]
    res_sections = {section: "" for section in sections}
    try:
        res = ask_llm(messages=messages, model=model)[-1]["content"]
    except Exception as e:
        print(colored(f"Error with LLM: {e}", "red"))
        return res_sections
    for section in sections:
        if res.find(f"<{section}>") != -1:
            res_sections[section] = deindent(
                res[
                    res.find(f"<{section}>") + len(f"<{section}>") : res.find(
                        f"</{section}>"
                    )
                ]
            )
        else:
            res_sections[section] = ""

    return res_sections


def gen_sections(
    text: str | list[str],
    sections: list[str] = SECTIONS,
    model: str = ModelName.GEMINI,
    sep: str = SEP,
    existing_sections: dict[str, list[str] | str] = None,
) -> dict[str, str]:
    if isinstance(text, str):
        text = [text]
    text = flatten_list(text)
    if sep:
        extra_seps = [sep[0] + "/" + sep[1:], sep[:-1] + "/" + sep[-1]]
    else:
        extra_seps = []
    res_sections = {section: "" for section in sections}
    if existing_sections:
        for k, v in existing_sections.items():
            if isinstance(v, list):
                v = "\n".join([str(val) for val in v if val])
            res_sections[k] = v
    for txt in text:
        res_sections_ = gen_sections_(txt, sections=sections, model=model, sep=sep)
        for section in sections:
            section_gen = res_sections_[section]
            if not section_gen or len(section_gen) <= 1:
                section_gen = ""
            res_sections[section] += section_gen + "\n"
            for sp in extra_seps:
                res_sections[section] = res_sections[section].replace(sp, sep)
    for k, v in res_sections.items():
        if sep:
            v = [deindent(v_.strip()) for v_ in v.split(sep=sep) if v_]
        else:
            v = [deindent(v.strip())]
        if v == [""]:
            v = []
        res_sections[k] = v
    return res_sections


def post_llm_list(section_text: list[str]):
    try:
        return flatten_list(
            [
                [
                    x.strip()
                    for x in flatten_list(
                        [s.split("\n") for s in flatten_list(res.split(","))]
                    )
                    if "i don't know" not in x.lower()
                ]
                for res in section_text
            ]
        )
    except Exception as e:
        print(colored(f"Error in post_llm_list: {e}", "red"))
        return section_text


def gen_ner_sections(text: str | list[str]) -> dict[str, list[str]]:
    result = {}
    if isinstance(text, list):
        text = " ".join(flatten_list(text))
    try:
        ner = gen_sections(
            text=text,
            sections=[
                "Companies worked at (Just the names. No extra info)",
                "Job Titles (Just the titles. No extra info)",
                "Skills",
                "Certifications (Just the names. No extra info)",
                "Educational Institutions (Just the names. No extra info)",
                "Highest Degree (Just the degree. No extra info)",
                "Full Name (Just the name. No extra info)",
                "Email",
                "Phone Number",
                "Location",
            ],
        )
        name = ner["Full Name (Just the name. No extra info)"][0]
        if name:
            first_name, last_name = split_name(name)
        else:
            first_name, last_name = "", ""
        # print(colored(f"\n\nNER: {ner}\n\n", "cyan"))
        try:
            email = ner["Email"][0]
            email = email.replace(" com", ".com")
            email = re.sub(r"[\s]+", "", email)
            email = [email]
        except Exception:
            email = ner["Email"]
        result["name"] = ner["Full Name (Just the name. No extra info)"]
        result["first_name"] = [first_name]
        result["last_name"] = [last_name]
        result["email"] = email
        result["phone"] = ner["Phone Number"]
        result["location"] = ner["Location"]
        result["edu_institutions"] = post_llm_list(
            ner["Educational Institutions (Just the names. No extra info)"]
        )
        result["highest_degree"] = ner[
            "Highest Degree (Just the degree. No extra info)"
        ]
        result["companies"] = post_llm_list(
            ner["Companies worked at (Just the names. No extra info)"]
        )
        result["titles"] = post_llm_list(
            ner["Job Titles (Just the titles. No extra info)"]
        )
        result["skills"] = post_llm_list(ner["Skills"])
        result["certifications"] = post_llm_list(
            ner["Certifications (Just the names. No extra info)"]
        )
        try:
            result = {
                k: [item.replace("Not Provided", "") for item in v if item]
                for k, v in result.items()
            }
        except Exception:
            pass
    except Exception as e:
        print(colored(f"Coud not generate NER sections: {e}", "red"))
    return result


def write_df_sections(
    row: pd.Series,
    sections_col: str = "sections",
    sections_folder: str = "sections",
    remote_sections_folder: str = "",
) -> pd.Series:
    if not sections_folder:
        return row
    file_path = Path(row["path"]).stem
    sections = row[sections_col]
    with open(f"{sections_folder}/{file_path}.json", "w") as f:
        json.dump(sections, f)
    if remote_sections_folder and is_bucket(remote_sections_folder):
        bucket_up(
            local_folder=str(sections_folder),
            bucket=str(remote_sections_folder),
            only_new=False,
        )
    return row


# def gen_list_of_sections(
#     text: str | list[str],
#     sections: list[str] = JOBS_SECTIONS,
#     mini_sections: list[str] = JOBS_MINI_SECTIONS,
#     main_section: str = "Company Names",
#     model: str = ModelName.GEMINI,
#     sep: str = SEP,
# ) -> list[dict[str, str]]:
#     if not main_section:
#         main_section = sections[0]
#     assert main_section in sections, f"{main_section} not in {sections}"
#     if mini_sections:
#         assert len(sections) == len(
#             mini_sections
#         ), "sections and mini_sections must have the same length"
#     else:
#         mini_sections = sections
#     sections_map = {sec: mini_sec for sec, mini_sec in zip(sections, mini_sections)}
#     res_sections = gen_sections(text, sections=sections, model=model, sep=sep)
#     sections_list = []
#     for i in range(len(res_sections[main_section])):
#         section = {}
#         for k, v in res_sections.items():
#             try:
#                 section[sections_map[k]] = v[i]
#             except IndexError:
#                 section[sections_map[k]] = ""
#         sections_list.append(section)
#     return sections_list


# def gen_embeddings_st(
#     text: str | list[str], model: str = EMS_MODEL, device: str = "cuda"
# ) -> list[float]:
#     if not isinstance(text, list):
#         text = [text]
#     ems_model = SentenceTransformer(model_name_or_path=model, device=device)
#     ems = ems_model.encode(text).tolist()
#     return ems


def gen_embeddings_google(
    text: str | list[str], model: str = GOOGLE_EMS_MODEL
) -> list[float | list[float]]:
    if not isinstance(text, list):
        text = [text]
    try:
        ems = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document",
            title="Embedding of sections",
        )["embedding"]
    except Exception as e:
        print(colored(f"\n\nCould not generate embeddings: {e}\n\n", "red"))
        ems = []
    return ems


def text_to_dict(
    text: str | list[str], key: str = "text"
) -> dict[str, list[str] | str]:
    if not isinstance(text, dict):
        if not isinstance(text, list):
            text = [text]
        text = {key: text}
    return text


def write_df_ems(
    row: pd.Series,
    ems_col: str = "embeddings",
    text_col: str = "sections",
    ems_folder: str = "ems",
    remote_ems_folder: str = "",
) -> pd.Series:
    if not ems_folder:
        return row
    file_path = Path(row["path"]).stem
    text_col = text_col if text_col in row.index else "text"
    text_dict = row[text_col]
    text_dict = text_to_dict(text_dict)
    ems_dict: dict = row[ems_col]
    for section, ems in ems_dict.items():
        if not ems:
            ems = [[]]
        for i in range(len(ems)):
            if not text_dict[section]:
                text = []
            else:
                text = text_dict[section][i]
            text_ems = ems[i]
            json_file = f"{str(ems_folder)}/{file_path}_{section}_{i+1}.json"
            json_ems = {
                "id": f"{file_path}_{section}_{i+1}",
                "restricts": [
                    {
                        "namespace": "tenant_id",
                        "allow": [row.get("tenant_id", "ten_1234")],
                    }
                ],
                "paragraph": text,
                "embedding": text_ems,
            }
            # print(colored(f"Writing to {json_file}", "cyan"))
            # print(json_ems)
            with open(json_file, "w") as f:
                json.dump(json_ems, f)
    if remote_ems_folder and is_bucket(remote_ems_folder):
        bucket_up(
            local_folder=str(ems_folder), bucket=str(remote_ems_folder), only_new=False
        )
    return row


def gen_embeddings(
    text: dict[str, list[str] | str] | str | list[str],
    embedding_fn: Callable = gen_embeddings_google,
    mean: bool = False,
) -> dict[str, list[list[float]]]:
    text = text_to_dict(text)
    ems_dict = {section: [] for section in text.keys()}
    for section, section_texts in text.items():
        if section_texts:  # and section_texts != [""]:
            try:
                ems_dict[section] = embedding_fn(section_texts)
            except Exception as e:
                print(colored(f"\n\nCould not generate embeddings: {e}\n\n", "red"))
    if mean:
        ems_dict = {
            section: np.mean(em, axis=0).tolist()
            for section, em in ems_dict.items()
            if em
        }
    return ems_dict


# def gen_embeddings(
#     text: dict[str, list[str] | str] | str | list[str],
#     chunk_size: int | None = CHUNK_SIZE,
#     chunk_overlap: int = CHUNK_OVERLAP,
#     embedding_fn: Callable = gen_embeddings_google,
#     mean: bool = False,
# ) -> dict[str, list[list[float]]]:
#     if not isinstance(text, dict):
#         text = {"text": text}
#     section_names = []
#     section_texts = []
#     for k, v in text.items():
#         if v:
#             if not isinstance(v, list):
#                 v = [v]
#             v = [deindent(val) for val in flatten_list(v) if val]
#             if not v:
#                 continue
#             if chunk_size is not None:
#                 v = load_and_split_resume_text(
#                     text=v, chunk_size=chunk_size, chunk_overlap=chunk_overlap
#                 )
#             section_texts += v
#             section_names += [k] * len(v)
#     # print(f"\n\nSection names: {section_names}\n\n")
#     # print(f"\n\nSection texts: {section_texts}\n\n")
#     ems_dict = {section: [] for section in text.keys()}
#     try:
#         ems = embedding_fn(section_texts)
#     except Exception as e:
#         print(colored(f"\n\nCould not generate embeddings: {e}\n\n", "red"))
#         return ems_dict
#     for section, em in zip(section_names, ems):
#         ems_dict[section].append(em)
#     if mean:
#         ems_dict = {
#             section: np.mean(em, axis=0).tolist() for section, em in ems_dict.items()
#         }
#     return ems_dict
