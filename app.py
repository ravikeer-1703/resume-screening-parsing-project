import PyPDF2
import numpy as np
import streamlit as st
import pandas as pd
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
nltk.download("stopwords")

from preprocessing import tfidf
from preprocessing import text_cleaning_func
from preprocessing import extract_skills
from preprocessing import extract_education
from preprocessing import email_id
from preprocessing import phone_number
from preprocessing import extracting_name
from preprocessing import extracting_work_exp

from preprocessing import extract_skills_str
from preprocessing import extract_edu_str

#===========================Page Configuration===============================================================

st.set_page_config(page_title="Resume Screening Services", layout="wide", page_icon="🎈")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Resume Screening And Parsing</h1>",unsafe_allow_html=True)
st.space()

resume_folder = st.file_uploader("Upload Resume folder", accept_multiple_files=True, type=["pdf"])
jd = st.text_area("Write job description here")

st.markdown("""<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 200px;
    font-size: 18px;}
</style>
""", unsafe_allow_html=True)
button = st.button("Submit", key=1)

#======================================== App ==========================================================================
text_ = []
if resume_folder and jd:
    for file in resume_folder:
        try:
            pdf_to_text = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_to_text.pages:
                text += page.extract_text()
            text_.append(text)
        except Exception as e:
            print(f"There is Error: {e}")

df = pd.DataFrame({"text": text_})

if button:
    # converting resume text into numerical value by using TF-IDF vectorizer
    clean_text = df["text"].apply(text_cleaning_func)
    tf = tfidf.fit_transform(clean_text).toarray()

    # converting job Discription text into numerical value by using TF-IDF vectorizer
    jd_cleaning = text_cleaning_func(jd)
    jd_tf = tfidf.transform([jd_cleaning]).toarray()

    #============================== Calculating cosine similarity ======================================================

    cs_similarity = np.round(cosine_similarity(tf,jd_tf)*100)
    final_df = pd.DataFrame(cs_similarity, columns=["Resume Score"]).sort_values(by = "Resume Score", ascending=False)
    final_df = final_df.reset_index(drop=True)


    # ================================ SKILLS EDUCATION MATCH SCORE =========================================================
    # ---------------------------------Skills match score calculation -------------------------------------------------------

    # Skills required as per job discription
    skills_in_jd = extract_skills(jd)
    skills_in_jd = [skills for skills in set(skills_in_jd.split())]
    def jd_skill_match_with_resume(text):
        matched = []
        for sk in skills_in_jd:
            if re.search(rf"\b{re.escape(sk)}\b", text):
                matched.append(sk)
        matched = list(set(matched))
        return " ".join(matched)

    def matching_skills_score(text):
        text = text.split()
        jd_skills_match = len(text)
        jd_num_skills = len(skills_in_jd)
        return np.round((jd_skills_match / jd_num_skills) * 100)

    # Job Discription skills match with candidates resume
    skills_matches = df["text"].apply(extract_skills_str)
    jd_skills_match = skills_matches.apply(jd_skill_match_with_resume)

    # Skills match score calculation function
    skills_match_score = jd_skills_match.apply(matching_skills_score)

    # ----------------------------------Education match score -----------------------------------------------------------

    # Education required as per Job Discription
    education_in_jd = extract_education(jd)
    education_in_jd = [x for x in education_in_jd.split()]


    def education_match(text):
        matched = []
        for sk in education_in_jd:
            if re.search(rf"\b{re.escape(sk)}\b", text):
                matched.append(sk)
        matched = list(set(matched))
        return " ".join(matched)


    def matching_education_score(text):
        text = text.split()
        matched_edu = len(text)
        jd_num_edu = len(education_in_jd)
        return np.round((matched_edu / jd_num_edu) * 100)


    # JD education match with candidates resume function
    edu_matches = df["text"].apply(extract_edu_str)
    education_match = edu_matches.apply(education_match)

    # Education matchin score calculation
    education_match_score = education_match.apply(matching_education_score)


    #=========================== RESUME PARSING DETAILS ================================================================
    # Details of Candidates extracting from resume add to dataframe name final_df
    final_df["skills_edu_match_score"] = (skills_match_score * 0.5) + (education_match_score * 0.5)
    final_df["name"] = df["text"].apply(extracting_name)
    final_df["phone"] = df["text"].apply(phone_number).str[0].fillna("")
    final_df["email"] = df["text"].apply(email_id).str[0].fillna("")
    final_df["work_exp"] = df["text"].apply(extracting_work_exp)
    final_df["skills"] = df["text"].apply(extract_skills)
    final_df["education"] = df["text"].apply(extract_education)

    st.space()
    st.dataframe(final_df.head(10))

    st.write("Number of Rows: ", final_df.shape[0])
    st.write("Number of Columns: ", final_df.shape[1])

#================================ CONNECTING TO DATABASE ===============================================================
    st.markdown("""<style>
        div.stButton > sql_button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 50px;
            width: 200px;
            font-size: 18px;}
        </style>
        """, unsafe_allow_html=True)
    engine = create_engine('mysql+mysqlconnector://root:ravi%4017031995@localhost/resume_database')
    sql_button = st.button("Data Load to Database", key=2)
    if sql_button:
        try:
            final_df.to_sql("resume_data", con=engine , if_exists="append", index=False)
            st.success("Data Load to Database Successfully")

        except Exception as e:
            st.error(f"There is Error: {e}")


