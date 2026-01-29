import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, TypedDict
import zipfile
import os
import pandas as pd
import pymupdf as fitz
from io import BytesIO
from langchain_core.output_parsers import CommaSeparatedListOutputParser

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("gemini")

st.title("AI-Powered Resume Analyzer & CSV Generator using LangChain")
upload_zip = st.file_uploader("Upload ZIP file", type=["zip"])

class DataFormat(TypedDict):
    name: str
    summary: str
    experience: Optional[int]
    skills: list[str]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
fm = model.with_structured_output(DataFormat)

if upload_zip:
    extract_dir = "extracted_file"
    if os.path.exists(extract_dir):
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    else:
        os.makedirs(extract_dir)

    with zipfile.ZipFile(BytesIO(upload_zip.read()), "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    st.success("ZIP file extracted successfully!")

    pdf_paths = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    if not pdf_paths:
        st.error("No PDF resumes found in ZIP")
        st.stop()

    all_results = []

    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        output = fm.invoke(text)

        all_results.append({
            "resume_file": os.path.basename(pdf_path),
            "name": output["name"],
            "summary": output["summary"],
            "experience": output["experience"],
            "skills": ", ".join(output["skills"])
        })

    df = pd.DataFrame(all_results)
    st.subheader("All Resume Results")
    st.dataframe(df)

    csv_text = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv_text,
        file_name="resume_results.csv",
        mime="text/csv"
    )