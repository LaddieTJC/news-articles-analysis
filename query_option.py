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


@st.experimental_memo
def googleNewsApi(query):
    googlenews = GoogleNews()
    googlenews.get_news(query)
    df =  googlenews.results()
    df = pd.DataFrame(df)
    df['link'] = df['link'].apply(lambda x: "https://"+x)
    # df['content'] = df['link'].apply(newspaper_3k)
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

# def displayNewsForNLP(df):
#     with st.form(df['link']):
#         st.header(f"[{df['title']}]({df['link']})")
#             # break
#         st.write(df['date'])
#         view = st.form_submit_button("Analysis")
#         if view:
#             st.session_state['r_article'] = df['title']
#             st.session_state['content'] = df.link.apply(newspaper_3k)

def displayNews(df):
    st.header(f"[{df['title']}]({df['link']})")
        # break
    st.write(df['date'])
# def clean_url(data):
#     html = urllib.request.urlopen(data).read().decode('utf-8')
#     text = get_text(html)
#     clean_url = extractor.find_urls(text)
#     return clean_url[0]


sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
)
NER = spacy.load("en_core_web_sm")
openai.api_key = "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"   
st.set_page_config(layout="wide") 
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

# if 'classifier' not in st.session_state:
#     classifier = pipeline("zero-shot-classification",
#                         model="facebook/bart-large-mnli")
#     st.session_state['classifier'] = classifier
with col1: 
    company_expander = st.expander(label='Company Filtering')
    articles=pd.DataFrame()
    with company_expander:
        # categories = ['Client News','Joint Venture','Layoffs and Staffing','Merger Acquisition','Partnership','C-suite','Product Launch']
        # # if 'article_df' not in st.session_state:
        #     # classifier = st.session_state['classifier']
        # st.selectbox("Select topic for company news:",categories)
        company = st.text_input("Search for company:")
        if company:
            with st.spinner("Please wait loading news"):
                articles = googleNewsApi(company)
    if type(articles) != 'str':
        articles.apply(displayNews,axis=1)
    # st.write(articles)
        # st.write(len(articles.results()))
    # for index,row in articles.iterrows():
    #     st.header(f"[{row['title']}]({row['link']})")
    #     # break
    #     st.write(row['date'])


with col2: 
    news_expander = st.expander(label="Search News")
    prompt = ""
    keyword_articles = []
    with news_expander:
        with st.form("related_news"):
            query = st.text_input("Insert keyword:")
            choice = st.number_input("number of related keywords to return:",min_value=2,help="Return number of keywords related to your input")
            submitted = st.form_submit_button("Submit")
    if submitted or 'keyword_df' not in st.session_state:
        prompt = f"Find {choice} related search terms based on " + query +" and return in ordered list"
                # bd_df = pd.read_csv('Data/cyber_query_1400.csv',nrows=100)
                # bd_df = bd_df.drop('Unnamed: 0',axis=1)
                # st.session_state['bd'] = bd_df


        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt+query,
        max_tokens=100)
        regex = r'\b[A-Za-z\s]+\b'
        keywords = re.findall(regex,response['choices'][0]['text'])
    # st.write(response)
    # st.write(type(response['choices'][0]['text']))
    # st.write(response['choices'][0]['text'])
    # st.write("after regex:",test)
        for i in keywords:
            keyword_articles.append(googleNewsApi(i))
        st.session_state['keyword_df'] = pd.concat(keyword_articles, ignore_index=True)
        for index, row in  st.session_state['keyword_df'].iterrows():
            with st.form(f"form{index}"):
                st.header(f"[{row['title']}]({row['link']})")
                view = st.form_submit_button("View")
                if view:
                    st.session_state['r_article'] = row['title']
            # final_df.apply(displayNewsForNLP,axis=1)
    else:
        for index, row in  st.session_state['keyword_df'].iterrows():
            with st.form(f"form{index}"):
                st.header(f"[{row['title']}]({row['link']})")
                view = st.form_submit_button("View")
                if view:
                    st.session_state['r_article'] = row['title']
        
            # final_df['content'] = final_df['link'].apply(newspaper_3k) 
        # st.write(response['choices'][0]['text'])
        # output = re.split('(?:^|, )\d{1,2}\. ',response['choices'][0]['text'])
        # print(output)
            #     # choices = response['choices'][0]['text']
        #     st.session_state['text'] = response['choices'][0]['text'][1:-1]
        #     st.session_state['text'] = list(set(re.findall(r"\w+",st.session_state['text'])))
        # st.write(response['choices'][0]['text'])
        # st.write(st.session_state['text'])
        # query = ' '.join(map(lambda x: 'Cyber ' + x,st.session_state['text']))
        # query = " ". join('(Cyber ' + word + ') || ' for word in st.session_state['text'])
        # st.write(query)
#         for index, row in bd_df.iterrows():
#             with st.form(f"form{index}"):
#                 st.header(f"[{row['title']}]({row['link']})")
#                 view = st.form_submit_button("View")
#                 if view:
#                     st.session_state['r_article'] = row['title']
# # with col2:
#     bd_df = pd.read_csv('Data/cyber_query_1400.csv',nrows=100)
#     bd_df = bd_df.drop('Unnamed: 0',axis=1)
#     st.session_state['bd'] = bd_df
#     last_page = len(bd_df) // 15    
    
#     if 'text' not in st.session_state:
#         response = openai.Completion.create(
#         model="text-davinci-003",
#         prompt='5 words that cyber security don"t want to hear and separated by space\n',
#         max_tokens=100)
#         # choices = response['choices'][0]['text']
#         st.session_state['text'] = response['choices'][0]['text'][1:-1]
#         st.session_state['text'] = list(set(re.findall(r"\w+",st.session_state['text'])))
#     # st.write(response['choices'][0]['text'])
#     # st.write(st.session_state['text'])
#     # query = ' '.join(map(lambda x: 'Cyber ' + x,st.session_state['text']))
#     query = " ". join('(Cyber ' + word + ') || ' for word in st.session_state['text'])
#     st.write(query)
#     for index, row in bd_df.iterrows():
#         with st.form(f"form{index}"):
#             st.header(f"[{row['title']}]({row['link']})")
#             view = st.form_submit_button("View")
#             if view:
#                 st.session_state['r_article'] = row['title']


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
            
            
        st.subheader("Sentiment:")   
        st.write(sentiment(st.session_state['r_article'])[0]['label'])
        
        # st.write(bd_df[bd_df['title'] == st.session_state['r_article']]['summary'].values[0])
        # summary = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['summary'].values[0]
        with st.spinner("Summarizing.... Please wait"):
            summary  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['link'].values[0]
            st.subheader("Summary of article:")
            st.write(summarize(newspaper_3k(summary)))
  