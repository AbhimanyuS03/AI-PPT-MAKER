from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent
from tavily import TavilyClient
import pytesseract as pyt
import streamlit as st
import os
import time
from PIL import Image
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
print("MODULE LOADED SUCCESSFULLY : THANK YOU ")

# to show web app : complete page layout
st.set_page_config(layout="wide")

# to give title 
st.title("AI RESUME GENERATOR")
st.write("""THIS APP HELP TO CUSTOMISED PROFFESSIONAL RESUME
WITH LATEST JOB APPL LINKS""")

st.image("bg.png")

st.sidebar.title("fill important detailed which we required")
st.sidebar.image("bg.png")


# Step 3
Groq_AI_APIs_Keys = st.sidebar.text_input("Groq-API",type="password")
tavily_api_key =  st.sidebar.text_input("Tavily-API",type="password")
GOOGLE_API_KEYS = st.sidebar.text_input("Google-API",type="password")

all_API = [tavily_api_key, Groq_AI_APIs_Keys,
           GOOGLE_API_KEYS]
if not all(all_API):
    st.error("Must give API KEYS")
    st.stop()
elif all(all_API):
    st.success("API KEYS LOADED SUCCESSFULLY")
else:
    st.info("passes ALL the API keys Succesfully")

# Multiselect option
options=["Delhi","Mumbai",
         "pune","banglore",
         "gurugram"]
locations = st.sidebar.multiselect("Select Locations",
                                   options = options)

profile_op = ["Data Analysts","AI Engineer",
              "Gen AI developer", "Full-stack dev",
              "Data Scientist"]

profile = st.sidebar.multiselect("select Job Profile",
                                options = options)

# ===============GET USER INFO================== #
st.markdown("""### GET USER INFO ###""")
user_info = st.text_area(""" write you resume Decsriptions : """) 

# project flow
# model
# tool
# app.py

model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    google_api_key = GOOGLE_API_KEYS
)
# response=model.invoke("hello buddy")
# print(response.content)

def search_latest_news_jobs(query):
  """This Function Helps to Featch Latest news
  or jobs related article using tavily"""

  client = TavilyClient(
      api_key = tavily_api_key
  )
  response =  client.search(query)
  return response

# agent_creation
agent =  create_agent(
    model= model,
    tools = [search_latest_news_jobs]
)
# agent

def main_agent(agent,query):
  """this is main agent, or leader agents
  orchestrate sub agents """
  # giving prompt to create detaoled prompt
  # for code genrations

  prompt = """you are AI assistant and below
  given is a prompt, your
  task is give detailed prompt for tasks
  this
  you are a professional resume generator where
  user give there personal info
  you have t6o create detailed resume
  for studnets or proffesstional one
  , it must be in dynamic UI  and
  UX and with advanced css professional
  designning output in html
  formatting no  markdown allowed
  """

  response = agent.invoke({"messages":[{"role":"user","content": prompt}]})


  detailed_prompt = response["messages"][-1].content[-1]["text"]

  # save Prompt using file handling

  with open ("prompt.txt","w") as f:
    f.write(detailed_prompt)

  user_detailed = f""" below Given is user detailed genrate
  resume based on that , if not given below keep: default resume:
  python developer user detailed: {query}"""

  final_prompt = prompt + detailed_prompt + user_detailed

  #Code genrations
  response = agent.invoke({"messages":[{"role":"user","content": final_prompt}]})

  code = response["messages"][-1].content[-1]["text"]
  return code

# code = main_agent(agent,'HARJI MEHTA , GEN AI EXPERT')
# from IPython import display as DISPLAY
# DISPLAY.HTML(code)

  # fetch latest domain related jobs using tavily
def get_jobs(agent,Location = "Delhi", Profile = "Data Entry, Ml in Pyhton"):
  location ="Delhi"
  Profile= "Data entry, Ml in Pyhton"

  prompt = f"""based on user given job profile,
  fetch latest jobs or jobs apply article
  using naukri, linkdin, indeed or all populer
  job apply platform, show results with
  job profiles name, location, salary, company name,
  show jobs only related to given
  {location} and {profile}. output must be in
  professional HTML , naukri themes card with dynamic designs ,
  show atleast  top 10-20 results with direct apply link """

  response = agent.invoke({"messages":[{"role":"user",
                                        "content": prompt}]})
  code = response["messages"][-1].content[-1]["text"]

  return code

 # code = get_jobs(agent)
 # DISPLAY.HTML(code)

if st.button ("Genrate Resume"):
           with st.spinner("Agents running"):
                      code = main_agent(agent,user_info)
                      st.html(code, width="stretch",
                                  unsafe_allow_javascript=True)
                      st.divider()
                      jobe_code=get_jobs(agent,location,profile)
                      st,html(jobe_code, width="stretch",
                                 unsafe_allow_javascript=True)

