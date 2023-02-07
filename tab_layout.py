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
from datetime import datetime, timedelta

nltk.download('punkt')
st.set_page_config(layout="wide") 
kw_model = KeyBERT()
extractor = URLExtract()
openai.api_key = st.secrets['openai_api']


@st.cache(allow_output_mutation=True)
def runModel():

    """ Load Model """

    model = EncoderDecoderModel.from_pretrained("patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    NER = spacy.load("en_core_web_sm")
    return model,tokenizer,NER


# def googleNewsApi(query,fromDate= (datetime.today() - timedelta(days=364)).strftime("%m/%d/%Y"),toDate=datetime.today().strftime("%m/%d/%Y")):
    
#     """ Run GoogleNews Library"""

#     googlenews = GoogleNews(start=fromDate,end=toDate)
#     googlenews.get_news(query)
#     df = pd.DataFrame(googlenews.results())
#     if not df.empty:
#         df['link'] = df['link'].apply(lambda x: "https://"+x)
#     return df


def checkAndExtract(word):
    if re.search('[\u4e00-\u9fff]+',word):
        chinese_word = []
        chinese_word += re.findall(re.compile(u'[\u4e00-\u9fff]+'), word) 
        word = re.findall(re.compile(u'[a-zA-Z]+'), word)
        return " ".join(word),"".join(chinese_word)
    return word,None

def googleNewsApi(query,fromDate= (datetime.today() - timedelta(days=364)).strftime("%m/%d/%Y"),toDate=datetime.today().strftime("%m/%d/%Y")):
    
    """ Run GoogleNews Library"""
    
    query,cn_query = checkAndExtract(query)
    if query:
        googlenews = GoogleNews(start=fromDate,end=toDate)
        googlenews.get_news(query)
        df = pd.DataFrame(googlenews.results())
        if not df.empty:
            df['link'] = df['link'].apply(lambda x: "https://"+x)
        googlenews.clear()
    if cn_query:
        googlenews = GoogleNews(start=fromDate,end=toDate,lang='zh-cn')
        googlenews.get_news(cn_query)
        df = pd.concat([df,pd.DataFrame(googlenews.results())],ignore_index=True)
        if not df.empty:
            df['link'] = df['link'].apply(lambda x: "https://"+x)
    return df



def displayNews(df):
    st.header(f"[{df['title']}]({df['link']})")
    st.write(df['date'])
    st.write("Link: ",df['link'])


def summarize(text):

    """ Summarizing of article """

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def tokenizeForSummarizer(summary):

    """ Check to see if an article exceed 4096 tokens """

    if len(nltk.word_tokenize(summary)) > 4096:
        summary = " ".join(nltk.word_tokenize(summary)[40:3500])
    return summary

def newspaper_3k(data):

    """ Newspaper3k to scrape the news using GoogleNews Library """
    
    content = Article(data)
    try:
        content.download()
        content.parse()
        return content.text
    except Exception as e:
        return False
def companies_list():
    df = pd.read_excel("vpc-list-forsmu.xlsx")
    return df['Company/Service Name'].tolist()

def del_session_state():
    if st.session_state:
        for key in st.session_state.keys():
            del st.session_state[key]

@st.cache
def analyzeArticle():
    st.session_state['summary'] = summarize(tokenizeForSummarizer(st.session_state['content']))
    st.session_state['keywords'] = ", ".join([i[0] for i in kw_model.extract_keywords(st.session_state['content'])])
    st.session_state['bi-keywords'] = ", ".join(i[0] for i in kw_model.extract_keywords(st.session_state['content'],keyphrase_ngram_range=(2,2)))
    st.session_state['entities'] = NER(st.session_state['content'])
    st.session_state['entities'] = list(set((e.label_,e.text) for e in st.session_state['entities'].ents))

model,tokenizer, NER = runModel()
comparables_dict  = {
    'Allay Therapeutics': ['Vertex Pharmaceuticals'],
    'EndoGastric Solutions (EGS)': ['GI Dynamics'],
    'Indapta Therapeutics':['Adicet Bio'],
    'AnHeart Therapeutics':['Ocugen'],
    'ArrayComm':['SkyCross '],
    'Binhui Biotech 武汉滨会生物科技股份有限公司 ':['BeiGene'],
    'CARD BioSciences Inc. (Regor)':['Vir Biotechnology'],
    '进化半导体 (Evolusia)':[''],
    '雅光丝琳控股(广东)有限公司 (Yaguang)':[''],
    'Neuspera Medical':['Neuromonics'],
    'Sonoma Biotherapeutics':['NexImmune'],
    'Amplication':['Flowingly'],
    'Bites Learning ':['Connecteam'],
    'Blings.IO':['Retrieve'],
    'EverAfter':['TidalScale'],
    'Firefly (aka Infralight)':['Lightlytics'],
    'GrowthSpace':['Mentorloop'],
    'Joonko':['Pymetrics'],
    'Navina':['Infervision'],
    'Sayata':['ErisX'],
    'Coinomo':['DTCO'],
    'Dedoco':['Eloops'],
    'Kapiva':['Colugo'],
    'Karkhana.io':['polySpectra'],
    'Onato':['eFishery'],
    'Pace':['Cashew Payments'],
    'IVF Access':['Healthwire'],
    'RPG Commerce':['SmartSites'],
    'SCB Abacus':['JupiterOne'],
    'Signzy':['Lightico'],   
    'Speedoc':['CiverMed'],
    'Threado':['Bazaarvoice'],
    'TipTip':['Embibe'],
    'Tortoise':['Hubble'],
    'Elotl':['Cloudigy '],
    'Cymulate':['Sophos', 'Crowdstrike', 'wiz', 'scrut', 'kenna security', 'attackiq'],
    'Evisort':['SirionOne', 'Icertis', 'Jaggaer', 'LinkSquares', 'Coupa', 'Agiloft'],
    'EasySend':['PandaDoc', 'SurveySparrow', 'eversign', 'DocuSign', 'Monday.com'],
    'Experify': ['Yotpo','Bazaarvoice','Trustpilot'],
    'Gitpod': ['Salesforce','AWS','Kinsta'],
    'Lumigo':['Datadog','Dynatrace','New Relic'],
    'Metaview':['SeekOut','TestGorilla','CodeSignal'],
    'OpsLevel':['Dynatrace','LogicMonitor','AppDynamics'],
    'Orkes':['Lightlytics','Knapsack','Dagger'],
    'Pronto':['Slack','Webex App','Google Workspace'],
    'Upsolver':['Snowflake','Posit','Qubole'],
    # 'Barramundi Group':[''],
    'ISE Foods Inc':['Shin-Shin Foods','DooleBoB','S Foods'],
    'InterOpera':['Kenovate Solutions','InterWeb','OLS software'],
    'Nuritas':['LabGenius','Brightseed','Metanovas']
}


def main():
    company_tab, bd_tab = st.tabs(["Company", 'Business Development'])

    # Portfolio News Tab  

    with company_tab:
        company_col, comparables_col = st.columns(2,gap='large')
        # Company News Column
        with company_col:
            company_expander = st.expander(label='Company Filtering')
            articles=pd.DataFrame()
            categories = ['All', 'Management Change','Use of Funds','C-Suite hiring','Fundraising','Layoffs and Staffing','Merger Acquisition']
            with company_expander:
                # company = st.selectbox("Select Company",list(comparables_dict.keys()))
                company = st.selectbox("Select Company",companies_list())
                news_class = st.selectbox('Categories:',categories)
                fromDate = (st.date_input("Date From",datetime.today() - timedelta(days=365))).strftime("%m/%d/%Y")
                toDate = (st.date_input("To")).strftime("%m/%d/%Y")
                if company or toDate or fromDate:
                    articles = googleNewsApi(company,fromDate=fromDate,toDate=toDate)
                    #Comparable News Column 
                    with comparables_col:
                        st.subheader("Competitor News:")
                        comparables_list = comparables_dict[company]
                        st.session_state['com_df'] = pd.concat((googleNewsApi(i) for i in comparables_list), ignore_index=True)
                        if not st.session_state['com_df'].empty:
                            st.session_state['com_df'].apply(displayNews,axis=1)
                        else:
                            st.write("No competitors news retrieved.")
            st.subheader("Company News:")
            if not articles.empty:
                articles.apply(displayNews,axis=1)
            else:
                st.write("No news retrieved")

    # Business Development Tab 

    with bd_tab:
        bd_col, nlp_col = st.columns(2)
        # Show news based on user query and openAI keywords 
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
                keywords = [query] + re.findall(regex,response['choices'][0]['text'])
                st.session_state['keyword_df'] = pd.concat((googleNewsApi(i) for i in keywords), ignore_index=True)
                st.write(keywords)
            if "keyword_df" in st.session_state:
                for index, row in  st.session_state['keyword_df'].iterrows():
                    st.header(f"[{row['title']}]({row['link']})")
                    st.write(row['date'])
                    st.write('link: ',row['link'])
                    view = st.button("View",key=index)
                    if view:
                        # del_session_state()
                        st.session_state['r_article'] = row['title']
                        st.session_state['has_article'] = True
            else:
                st.write("Search for news")

        # Column for NLP  
        with nlp_col:
            if 'r_article' in st.session_state:
                st.session_state['content']  = st.session_state['keyword_df'][st.session_state['keyword_df']['title'] == st.session_state['r_article']]['link'].values[0]
                st.session_state['content'] = newspaper_3k(st.session_state['content'])

                if st.session_state['content']:
                    if st.session_state['has_article'] == True:
                        with st.spinner("Analyzing..."):
                            # analyzeArticle()
                            st.session_state['summary'] = summarize(tokenizeForSummarizer(st.session_state['content']))
                            st.session_state['keywords'] = ", ".join([i[0] for i in kw_model.extract_keywords(st.session_state['content'])])
                            st.session_state['bi-keywords'] = ", ".join(i[0] for i in kw_model.extract_keywords(st.session_state['content'],keyphrase_ngram_range=(2,2)))
                            st.session_state['entities'] = NER(st.session_state['content'])
                            st.session_state['entities'] = list(set((e.label_,e.text) for e in st.session_state['entities'].ents))
                        st.session_state['has_article'] = False
                    st.header("Summary of article:")
                    st.write(st.session_state['summary'])
                    st.header("Top 5 keywords from article: ")
                    st.subheader("Unigram:")
                    st.write(st.session_state['keywords'] )
                    st.subheader("Bigram:")
                    st.write(st.session_state['bi-keywords'])
                    st.session_state['ent_type'] = st.selectbox("Filter entities:",NER.get_pipe("ner").labels)
                    st.session_state['ent_df'] = pd.DataFrame(st.session_state['entities'] , columns=['Entity','Identified']).sort_values('Entity')
                    if st.session_state['ent_type']:
                        st.table(st.session_state['ent_df'][st.session_state['ent_df']['Entity'] == st.session_state['ent_type']])
                else:
                    st.write("Content cannot be retreived")



if __name__ == "__main__":
    main()