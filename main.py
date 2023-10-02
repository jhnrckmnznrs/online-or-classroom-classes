import numpy as np
import pandas as pd

import streamlit as st

from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
# add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("main.py", "Introduction"),
        Page("pages/dataframe.py", "Enrollment Data"),
        Page("pages/regression.py", "Regression Model"),
    ]
)

st.title("Online vs. Classroom: Which Enrollment Type is Right?")

st.markdown("""
            <h2> Background and Objective </h2>
            <div style="text-align: justify;">
            You are working as a data scientist at a local University. The university started offering online courses to reach a wider range of students. The university wants you to help them understand enrollment trends. They would like you to identify what contributes to higher enrollment. In particular, whether the course type (online or classroom) is a factor.
            </div>
            <h2> Introduction </h2>
            <div style="text-align: justify;">
            Enrollment trends in online courses accelerated dramatically during the COVID-19 pandemic. During the height of the pandemic, most classes moved to online-only instruction. While the effects of the pandemic has subsided, online learning remains popular, with many students choosing to take online courses or even complete entire degrees online. In hindsight, online courses offer flexibility and convenience that traditional classroom-based courses cannot. Students can take online courses at their own pace and on their own schedule, from anywhere in the world. This makes online learning a good option for students who are working full-time, have families, or live in remote areas. However, there are also some drawbacks such as the difficulty of staying motivated and engaged. Students also miss out on the social interaction and networking opportunities that are available in traditional classroom settings. It is important to note that online and onsite learning are not mutually exclusive. Many students choose to take a mix of online and onsite courses. This allows them to take advantage of the benefits of both types of learning.
            </div>
            """, unsafe_allow_html=True
            )