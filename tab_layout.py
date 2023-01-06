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
from keybert import KeyBERT


st.set_page_config(layout="wide") 
kw_model = KeyBERT()
extractor = URLExtract()
# model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# sentiment = pipeline(
#     "sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english",
#     tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
# )
# NER = spacy.load("en_core_web_sm")
openai.api_key = "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"   


@st.experimental_memo
def runModel():
    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    )
    NER = spacy.load("en_core_web_sm")
    return model,tokenizer,sentiment,NER

@st.experimental_memo
def googleNewsApi(query):
    googlenews = GoogleNews()
    googlenews.get_news(query)
    df =  googlenews.results()
    df = pd.DataFrame(df)
    df['link'] = df['link'].apply(lambda x: "https://"+x)
    return df

def displayNews(df):
    st.header(f"[{df['title']}]({df['link']})")
    st.write(df['date'])


def summarize(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    # Get the summary from the output tokens
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def articleNLP(title,col):
    with col:
        st.session_state['summary']  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == title]['link'].values[0]
        st.session_state['summary'] = newspaper_3k(st.session_state['summary'])
        if st.session_state['summary']:
                with st.spinner("Analyzing..."):
                    st.subheader("Summary of article:")
                    st.write(summarize(tokenizeForSummarizer(st.session_state['summary'])))
                    st.subheader("Sentiment:")   
                    st.write(sentiment(title)[0]['label'])
                    st.subheader("Top 5 keywords from article: ")
                    st.write(", ".join([i[0] for i in kw_model.extract_keywords(st.session_state['summary'])]))
                    entity = NER(st.session_state['summary'])
                    entities = set((e.label_,e.text) for e in entity.ents)
                    # print(entities)
                    entities =list(entities)
                    st.session_state['ent_type'] = st.selectbox("Filter entities:",NER.get_pipe("ner").labels)
                    st.session_state['ent_df'] = pd.DataFrame(entities, columns=['Entity','Identified']).sort_values('Entity')
                    st.table(st.session_state['ent_df'])
                    if st.session_state['ent_type']:
                        st.table(st.session_state['ent_df'][st.session_state['ent_df']['Entity'] == st.session_state['ent_type']])
                    
        else:
            st.write("Cannot retrieve article")


def tokenizeForSummarizer(summary):
    if len(nltk.word_tokenize(summary)) > 4096:
        summary = " ".join(nltk.word_tokenize(summary)[40:3500])
    return summary

def newspaper_3k(data):
    content = Article(data)
    try:
        content.download()
        content.parse()
        return content.text
    except Exception as e:
        return False
model,tokenizer, sentiment, NER = runModel()
comparables_dict  = {
    'cymulate':['Sophos', 'Crowdstrike', 'wiz', 'scrut', 'kenna security', 'attackiq'],
    'evisort':['SirionOne', 'Icertis', 'Jaggaer', 'LinkSquares', 'Coupa', 'Agiloft'],
    'easysend':['PandaDoc', 'SurveySparrow', 'eversign', 'DocuSign', 'Monday.com'
]
}
def main():
    company_tab, bd_tab = st.tabs(["Company", 'Business Development'])
    with company_tab:
        company_col, comparables_col = st.columns(2,gap='large')
        with company_col:
            company_expander = st.expander(label='Company Filtering')
            articles=pd.DataFrame()
            categories = ['All', 'Partnership','Client News','C-Suite']
            with company_expander:
                # company = st.text_input("Search for company:")
                company = st.selectbox("Select Company",list(comparables_dict.keys()))
                news_class = st.selectbox('Categories:',categories)
                if company:
                    with st.spinner("Please wait loading news"):
                        articles = googleNewsApi(company)
            if type(articles) != 'str':
                articles.apply(displayNews,axis=1)
        with comparables_col:
            if company:
                comparables_list = comparables_dict[company.lower()]
                st.write(comparables_list)


    with bd_tab:
        bd_col, nlp_col = st.columns(2)
        with bd_col:
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
                    view = st.button("View",key=index)
                    if view:
                        st.session_state['r_article'] = row['title']
                        st.session_state['has_article'] = True
                    # view = st.button("View",key=index,on_click=articleNLP,args=(row['title'],nlp_col, ))
            else:
                st.write("Search for news")
        with nlp_col:
            if 'r_article' in st.session_state:
                st.session_state['content']  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['link'].values[0]
                st.session_state['content'] = newspaper_3k(st.session_state['content'])

                if st.session_state['content']:
                    if st.session_state['has_article'] == True:
                        with st.spinner("Analyzing..."):
                            st.session_state['summary'] = summarize(tokenizeForSummarizer(st.session_state['content']))
                            st.session_state['sentiment'] = sentiment(st.session_state['r_article'])[0]['label']
                            st.session_state['keywords'] = ", ".join([i[0] for i in kw_model.extract_keywords(st.session_state['content'])])
                            st.session_state['entities'] = NER(st.session_state['content'])
                            st.session_state['entities'] = list(set((e.label_,e.text) for e in st.session_state['entities'].ents))
                        st.session_state['has_article'] = False
                            
                    st.subheader("Summary of article:")
                    st.write(st.session_state['summary'])
                    st.subheader("Sentiment:")   
                    st.write(st.session_state['sentiment'])
                    st.subheader("Top 5 keywords from article: ")
                    st.write(st.session_state['keywords'] )
                    st.session_state['ent_type'] = st.selectbox("Filter entities:",NER.get_pipe("ner").labels)
                    st.session_state['ent_df'] = pd.DataFrame(st.session_state['entities'] , columns=['Entity','Identified']).sort_values('Entity')
                    if st.session_state['ent_type']:
                        st.table(st.session_state['ent_df'][st.session_state['ent_df']['Entity'] == st.session_state['ent_type']])
                else:
                    st.write("Content cannot be retreived")
        # with nlp_col:
            
        #     if 'r_article' in st.session_state:
        #         summary  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['link'].values[0]
        #         summary = newspaper_3k(summary)
        #         st.write(summary)
            # if summary:
            #         with st.spinner("Analyzing..."):
            #             st.subheader("Summary of article:")
            #             st.write(summarize(tokenizeForSummarizer(summary)))
            #             st.subheader("Sentiment:")   
            #             st.write(sentiment(st.session_state['r_article'])[0]['label'])
            #             entity = NER(summary)
            #             entities = set((e.label_,e.text) for e in entity.ents)
            #             # print(entities)
            #             entities =list(entities)
            #             st.table(pd.DataFrame(entities, columns=['Entity','Identified']))
            # else:
            #     st.write("Cannot retrieve article")


if __name__ == "__main__":
    main()