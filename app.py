# streamlit_app.py

import streamlit as st
import tempfile
import os
import json
from resume_parser import ResumeParser

def main():
    st.title("📄 Resume Parser App")
    st.write("Upload a resume (PDF) and get structured JSON data!")

    uploaded_file = st.file_uploader("Choose a resume PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        st.success("Resume uploaded successfully! 🔥")

        # Parse the resume
        parser = ResumeParser()
        with st.spinner("Parsing resume... ⏳"):
            result = parser.parse_resume(temp_path)

        # Remove the temp file
        os.remove(temp_path)

        # Display results
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("🎯 Parsed Resume JSON")
            st.json(result["parsed"])  # Showing only clean parsed result
            
            with st.expander("See Detailed Parsing with Confidence Scores"):
                st.json(result["detailed"])  # If you want to also show detailed info

if __name__ == "__main__":
    main()
