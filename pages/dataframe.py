import numpy as np
import pandas as pd

import pygwalker as pyg

import streamlit as st
import streamlit.components.v1 as components

filepath = "https://s3.amazonaws.com/talent-assets.datacamp.com/university_enrollment_2306.csv"
df = pd.read_csv(filepath)

values = {"course_type": 'classroom', "year": 2011, "enrollment_count": 0, "pre_score": '0', "post_score": 0, "pre_requirement": 'None', "department": 'unknown'}
df = df.fillna(value = values)
df['pre_score'] = df['pre_score'].replace('-', '0')
df['pre_score'] = df['pre_score'].astype(float)
df['department'] = df['department'].str.strip().replace('Math', 'Mathematics')

st.title('Enrollment Data')
st.markdown("""
            For each course uniquely idenfitied through its `course_id`, the [dataset](https://s3.amazonaws.com/talent-assets.datacamp.com/university_enrollment_2306.csv) include the following information: 
            
            1. Course Type (`course_type`)
            2. Year Offered (`year`)
            3. Number of Enrollees (`enrollment_count')
            4. Pre-Assessment Score (`pre_score`)
            5. Post-Assessment Score (`post_score`)
            6. Pre-Requisite (`pre_requirement`)
            6. Offering Department (`department`)
        
            """
            )

st.dataframe(df, hide_index = True, use_container_width = True)
st.markdown("""
            <div style="text-align: justify;">
            Note that the dataframe above is a cleaned version. The cleaning process include equating missing scores as 0 and labelling missing pre-requisites <code>None</code>, and replacing <code>Math</code> department with <code>Mathematics</code>.
            </div>
            """, unsafe_allow_html=True
            )
st.write("## Properties")
st.write('###### Dimension:', df.shape)
st.write('###### Unique Course Types:', df['course_type'].sort_values().unique().tolist())
st.write('###### Available Years:', df['year'].sort_values().unique().tolist())
st.write('###### Unique Pre-requisites:', df['pre_requirement'].sort_values().unique().tolist())
st.write('###### University Departments:', df['department'].sort_values().unique().tolist())
st.write('###### Summary Statistics')
st.dataframe(df.describe(), use_container_width = True)

st.write("## Visualization")

st.write("Visual explorations for this section are generated using [Pygwalker](https://docs.kanaries.net/pygwalker).")

pyg_html = pyg.walk(df, return_html=True)
 
# Embed the HTML into the Streamlit app
components.html(pyg_html, width=1000, height = 950, scrolling=True)
