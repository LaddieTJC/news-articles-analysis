import streamlit as st
import pandas as pd
import openai
import re
import spacy
from GoogleNews import GoogleNews
import numpy as np
from transformers import pipeline, EncoderDecoderModel, LongformerTokenizer
from urlextract import URLExtract
from newspaper import Article
extractor = URLExtract()
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
)
NER = spacy.load("en_core_web_sm")
openai.api_key = "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"   
st.set_page_config(layout="wide") 

@st.experimental_memo
def googleNewsApi(query):
    googlenews = GoogleNews()
    googlenews.get_news(query)
    df =  googlenews.results()
    df = pd.DataFrame(df)
    df['link'] = df['link'].apply(lambda x: "https://"+x)
    return df

@st.experimental_memo
def summarize(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    # Get the summary from the output tokens
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def newspaper_3k(data):
    content = Article(data)
    try:
        content.download()
        content.parse()
        return content.text
    except Exception as e:
        return False


def displayNews(df):
    st.header(f"[{df['title']}]({df['link']})")
    st.write(df['date'])




# def newsCatcherApi(company):
#     """News Catcher API"""
#     # ['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'cn', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'tw', 'uk', 'ur', 'vi']
#     newscatcherapi = NewsCatcherApiClient(x_api_key='7oYgxk-M9R3aSmswlP-LoEqMkmlYHmDdGEooCng2Ww4')
#     base_url = "https://api.newscatcherapi.com/v2/search"
#     headers = {'x-api-key':'7oYgxk-M9R3aSmswlP-LoEqMkmlYHmDdGEooCng2Ww4'}
#     all_articles = newscatcherapi.get_search(q = company,
#                                          lang='en',
#                                          page_size=100,
#                                             page=1,
#                                             from_= '2020/1/1')

#     articles = all_articles['articles']

#     return articles

col1,col2,col3=st.columns(3)


with col1: 
    company_expander = st.expander(label='Company Filtering')
    articles=pd.DataFrame()
    with company_expander:
        company = st.text_input("Search for company:")
        if company:
            with st.spinner("Please wait loading news"):
                articles = googleNewsApi(company)
    if type(articles) != 'str':
        articles.apply(displayNews,axis=1)


with col2: 
    news_expander = st.expander(label="Search News")
    keyword_articles = []
    with news_expander:
        with st.form("related_news"):
            query = st.text_input("Insert keyword:")
            choice = st.number_input("number of related keywords to return:",min_value=2,help="Return number of keywords related to your input")
            submitted = st.form_submit_button("Submit")
    if submitted:

        prompt = f"Find {choice} related search terms based on " + query +" and return in ordered list"

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt+query,
        max_tokens=100)
        regex = r'\b[A-Za-z\s]+\b'
        keywords = re.findall(regex,response['choices'][0]['text'])

        for i in keywords:
            keyword_articles.append(googleNewsApi(i))
        st.session_state['keyword_df'] = pd.concat(keyword_articles, ignore_index=True)
        
        
    if "keyword_df" in st.session_state:
        for index, row in  st.session_state['keyword_df'].iterrows():
            st.header(f"[{row['title']}]({row['link']})")
            view = st.button("View",key=index)
            if view:
                st.session_state['r_article'] = row['title']
    else:
        st.write("Search for news")
    # else:
    #     if 'keyword_df' in st.session_state:
    #         for index, row in  st.session_state['keyword_df'].iterrows():
    #             with st.form(f"form{index}"):
    #                 st.header(f"[{row['title']}]({row['link']})")
    #                 view = st.form_submit_button("View")
    #                 if view:
    #                     st.session_state['r_article'] = row['title']



with col3:
    entity_label = {}
    if 'r_article' not in st.session_state:
        st.write("Please select an article")
    else:
        st.write(st.session_state['r_article'])
        entity = NER(st.session_state['r_article'])
   
        for e in entity.ents:
            if e.label_ == 'ORG' or e.label_ == 'GPE':
                if e.label_ in entity_label:
                    entity_label[e.label_].append(e.text)
                else:
                    entity_label[e.label_] = e.text
        if not entity_label:
            st.write("No entity extracted")
        else:
            data = pd.DataFrame.from_dict([entity_label])
            st.write(data)
            
            
        st.subheader("Sentiment:")   
        st.write(sentiment(st.session_state['r_article'])[0]['label'])
        
        with st.spinner("Summarizing.... Please wait"):
            summary  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['link'].values[0]
            st.subheader("Summary of article:")
            st.write(summarize(newspaper_3k(summary)))
  