Hello,
everyone i am Ravi.
Project name: Resume Screening and parsing.
Short note about this project: This resume screening and parsing is an intelligent web application and it is built by using Python, Streamlit for web UI, and Machine Learning.
It automates the process of analyzing resumes, extracting candidate information, and ranking candidates based on their relevance to a given job description and skills and education matched with job description.
This project can help recruiters and HR professionals in save time, improve efficiency, and make data-driven hiring decisions very fast.

How to use it:
- step-1: Upload resume pdf files.
- step-2: write or paste job description.
- step-3: click on submit button.
After all these 3 step it will process the all pdf file and show you a data table in which you can see resume score, skills education match score, candidates name and other details.
Download all Data as csv file: just hover mouse cursor on table and you will see download sign on head of table on most right side of the table and as you click on this sign table automatically download.


Features of this project:
- Intractive UI built with Streamlit.
- we can upload multiple pdf file in one time.
- it will analyze the all pdf and give the key information about candidates like:
          - candidates resume score as pe job description.
          - candidates skills and education score of skills and educations match with skills and educations required as per job description.
          - candidates name
          - phone number
          - e-mail id
          - work experience
          - skills
          - educations
- it calculates candidates resume score by using:
          - TF-IDF vectorizer, and
          - cosine similarity function
- it will store parse details in mysql database management system just click on "data load to database"

Tools and libraries used in this project:
  - Steamlit:  for fronted UI
  - python: for backend
  - Machine Learning: for scikit-learn ( TF-IDF, cosine_similarity)
  - MYSQL: for database
  - NLP: for resume parsing ( Spacy, NLTK, Regular Expression)
  - Pandas: for data manupulation and collection
  - Numpy: for numberical calculations
  - PyPDF2: for convert pdf to text file
  - sqlalchemy: for making connection between python and MYSQL







