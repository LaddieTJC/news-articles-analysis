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
import nltk

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

def articleNLP(title):
    with col3:
        entity_label = {}
        summary  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == title]['link'].values[0]
        summary = newspaper_3k(summary)
        if summary:
                with st.spinner("Analyzing..."):
                    st.subheader("Summary of article:")
                    st.write(summarize(tokenizeForSummarizer(summary)))
                    st.subheader("Sentiment:")   
                    st.write(sentiment(title)[0]['label'])
                    entity = NER(summary)
                    entities = set((e.label_,e.text) for e in entity.ents)
                    # print(entities)
                    entities =list(entities)
                    st.table(pd.DataFrame(entities, columns=['Entity','Identified']))
        else:
            st.write("Cannot retrieve article")

def tokenizeForSummarizer(summary):
    if len(nltk.word_tokenize(summary)) > 4096:
        summary = " ".join(nltk.word_tokenize(summary)[40:3500])
    return summary

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
    categories = ['All', 'Partnership','Client News','C-Suite']
    with company_expander:
        company = st.text_input("Search for company:")
        news_class = st.selectbox('Categories:',categories)
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
        st.write(keywords)
        for i in keywords:
            keyword_articles.append(googleNewsApi(i))
        st.session_state['keyword_df'] = pd.concat(keyword_articles, ignore_index=True)
        
        
    if "keyword_df" in st.session_state:
        for index, row in  st.session_state['keyword_df'].iterrows():
            st.header(f"[{row['title']}]({row['link']})")
            view = st.button("View",key=index,on_click=articleNLP,args=(row['title'], ))
    else:
        st.write("Search for news")
