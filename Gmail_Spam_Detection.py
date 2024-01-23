# import streamlit as st
# import imaplib
# import email
# import traceback
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import LdaModel
# from gensim.corpora import Dictionary
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import base64

# # Function to read emails from Gmail
# def read_email_from_gmail(email_address, password):
#     ORG_EMAIL = "@gmail.com"
#     FROM_EMAIL = email_address + ORG_EMAIL
#     FROM_PWD = password
#     SMTP_SERVER = "imap.gmail.com"
#     SMTP_PORT = 993

#     dataset = []  # List to store email data

#     try:
#         mail = imaplib.IMAP4_SSL(SMTP_SERVER)
#         mail.login(FROM_EMAIL, FROM_PWD)
#         mail.select('inbox')

#         data = mail.search(None, 'ALL')
#         mail_ids = data[1]
#         id_list = mail_ids[0].split()
#         first_email_id = int(id_list[0])
#         latest_email_id = int(id_list[-1])

#         for i in range(latest_email_id, first_email_id, -1):
#             data = mail.fetch(str(i), '(RFC822)')
#             for response_part in data:
#                 arr = response_part[0]
#                 if isinstance(arr, tuple):
#                     msg = email.message_from_string(str(arr[1], 'utf-8'))
#                     email_subject = msg['subject']
#                     email_from = msg['from']

#                     # Extracting the full email text
#                     email_text = ''
#                     for part in msg.walk():
#                         if part.get_content_type() == 'text/plain':
#                             email_text = part.get_payload(decode=True).decode('utf-8', 'ignore')
#                             break

#                     dataset.append({
#                         'From': email_from,
#                         'Subject': email_subject,
#                         'Text': email_text
#                     })

#     except Exception as e:
#         traceback.print_exc()
#         print(str(e))

#     return dataset

# # Function for text preprocessing and LDA
# def preprocess_and_lda(df):
#     # Handle missing values in the 'Text' column
#     df['Text'].fillna('', inplace=True)

#     # Assume you have a 'Text' column in the dataset containing email text
#     documents = df['Text'].tolist()

#     # Text preprocessing
#     stop_words = set(stopwords.words('english'))
#     ps = PorterStemmer()

#     def preprocess_text(text):
#         if isinstance(text, str):  # Check if 'text' is a string
#             text = text.lower()
#             text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
#         return text

#     documents = [preprocess_text(doc) for doc in documents]

#     # Create a Bag-of-Words representation
#     vectorizer = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
#     X = vectorizer.fit_transform(documents)

#     # Create a Gensim Dictionary from the documents
#     gensim_dict = Dictionary([doc.split() for doc in documents])

#     # Create a Gensim Corpus
#     corpus = [gensim_dict.doc2bow(doc.split()) for doc in documents]

#     # Apply LDA
#     num_topics = 2  # You can adjust the number of topics based on your dataset
#     lda_model = LdaModel(corpus, num_topics=num_topics, id2word=gensim_dict, passes=15)

#     # Get the topics for each document
#     df['Topics'] = [lda_model.get_document_topics(gensim_dict.doc2bow(doc.split())) for doc in documents]

#     return df

# # Function to classify emails based on spam threshold
# def classify_emails(df, spam_threshold=0.8):
#     # Ensure 'Topics' column exists
#     if 'Topics' not in df.columns:
#         st.warning("Topics column not found. Please run text preprocessing and LDA.")
#         return df

#     df['Label'] = df['Topics'].apply(lambda topics: 'spam' if any(prob > spam_threshold for _, prob in topics) else 'ham')
#     return df

# # Streamlit app
# def main():
#     st.title("Email Classifier App")

#     # User input for Gmail credentials
#     email_address = st.text_input("Enter your Gmail address:")
#     password = st.text_input("Enter your Gmail password:", type="password")

#     # Button to start email classification
#     if st.button("Start Email Classification"):
#         if email_address and password:
#             # Read emails from Gmail
#             emails_dataset = read_email_from_gmail(email_address, password)

#             # Create DataFrame
#             df = pd.DataFrame(emails_dataset)
#             df.to_csv('emails_dataset.csv', index=False)

#             # Preprocess and apply LDA
#             df = preprocess_and_lda(df)

#             # Check if 'Topics' column exists
#             if 'Topics' not in df.columns:
#                 st.warning("Topics column not found. Please run text preprocessing and LDA.")
#                 return

#             # Classify emails based on spam threshold
#             df = classify_emails(df)

#             # Display the resulting dataset
#             st.subheader("Resulting Dataset:")
#             st.write(df)

#             # Display email topics
#             st.subheader("Email Topics:")
#             for index, row in df.iterrows():
#                 st.write(f"Email {index + 1} Topics: {row['Topics']}")

#             # Display email labels
#             st.subheader("Email Labels:")
#             for index, row in df.iterrows():
#                 st.write(f"Email {index + 1} - Label: {row['Label']}")

#             # Save the DataFrame with the new 'Label' column
#             df.to_csv('result_dataset.csv', index=False)

#             # Download link for the resulting dataset with labels
#             st.markdown("### Download Resulting Dataset with Labels")
#             st.write("Click below to download the CSV file.")
#             st.markdown(get_binary_file_downloader_html('result_dataset.csv', 'Result Dataset'), unsafe_allow_html=True)

#         else:
#             st.warning("Please enter your Gmail address and password.")

# # Function to create download link for a file
# def get_binary_file_downloader_html(file_path, file_label):
#     with open(file_path, 'rb') as f:
#         data = f.read()
#     b64 = base64.b64encode(data).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{file_label}.csv">Download {file_label}</a>'

# if __name__ == "__main__":
#     main()




import streamlit as st
import pandas as pd
import imaplib
import email
import traceback
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import base64
import matplotlib.pyplot as plt

# Set option to suppress PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to read emails from Gmail
def read_email_from_gmail(email_address, password):
    ORG_EMAIL = "@gmail.com"
    FROM_EMAIL = email_address + ORG_EMAIL
    FROM_PWD = password
    SMTP_SERVER = "imap.gmail.com"
    SMTP_PORT = 993

    dataset = []  # List to store email data

    try:
        mail = imaplib.IMAP4_SSL(SMTP_SERVER)
        mail.login(FROM_EMAIL, FROM_PWD)
        mail.select('inbox')

        data = mail.search(None, 'ALL')
        mail_ids = data[1]
        id_list = mail_ids[0].split()
        first_email_id = int(id_list[0])
        latest_email_id = int(id_list[-1])

        for i in range(latest_email_id, first_email_id, -1):
            data = mail.fetch(str(i), '(RFC822)')
            for response_part in data:
                arr = response_part[0]
                if isinstance(arr, tuple):
                    msg = email.message_from_string(str(arr[1], 'utf-8'))
                    email_subject = msg['subject']
                    email_from = msg['from']

                    # Extracting the full email text
                    email_text = ''
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            email_text = part.get_payload(decode=True).decode('utf-8', 'ignore')
                            break

                    dataset.append({
                        'From': email_from,
                        'Subject': email_subject,
                        'Text': email_text
                    })

    except Exception as e:
        traceback.print_exc()
        print(str(e))

    return dataset

# Function for text preprocessing and LDA
def preprocess_and_lda(df):
    # Handle missing values in the 'Text' column
    df['Text'].fillna('', inplace=True)

    # Assume you have a 'Text' column in the dataset containing email text
    documents = df['Text'].tolist()

    # Text preprocessing
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    def preprocess_text(text):
        if isinstance(text, str):  # Check if 'text' is a string
            text = text.lower()
            text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
        return text

    documents = [preprocess_text(doc) for doc in documents]

    # Create a Bag-of-Words representation
    vectorizer = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(documents)

    # Create a Gensim Dictionary from the documents
    gensim_dict = Dictionary([doc.split() for doc in documents])

    # Create a Gensim Corpus
    corpus = [gensim_dict.doc2bow(doc.split()) for doc in documents]

    # Apply LDA
    num_topics = 2  # You can adjust the number of topics based on your dataset
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=gensim_dict, passes=15)

    # Get the topics for each document
    df['Topics'] = [lda_model.get_document_topics(gensim_dict.doc2bow(doc.split())) for doc in documents]

    return df

# Function to classify emails based on spam threshold
def classify_emails(df, spam_threshold=0.8):
    # Ensure 'Topics' column exists
    if 'Topics' not in df.columns:
        st.warning("Topics column not found. Please run text preprocessing and LDA.")
        return df

    df['Label'] = df['Topics'].apply(lambda topics: 'spam' if any(prob > spam_threshold for _, prob in topics) else 'ham')
    return df

# Streamlit app
def main():
    st.title("Email Classifier App")

    # User input for Gmail credentials
    email_address = st.text_input("Enter your Gmail address:")
    password = st.text_input("Enter your Gmail password:", type="password")

    # Button to start email classification
    if st.button("Start Email Classification"):
        if email_address and password:
            # Read emails from Gmail
            emails_dataset = read_email_from_gmail(email_address, password)

            # Create DataFrame
            df = pd.DataFrame(emails_dataset)
            df.to_csv('emails_dataset.csv', index=False)

            # Preprocess and apply LDA
            df = preprocess_and_lda(df)

            # Check if 'Topics' column exists
            if 'Topics' not in df.columns:
                st.warning("Topics column not found. Please run text preprocessing and LDA.")
                return

            # Classify emails based on spam threshold
            df = classify_emails(df)

            # Display the resulting dataset
            st.subheader("Resulting Dataset:")
            st.write(df)

            # Display email topics
            st.subheader("Email Topics:")
            for index, row in df.iterrows():
                st.write(f"Email {index + 1} Topics: {row['Topics']}")

            # Display email labels
            st.subheader("Email Labels:")
            for index, row in df.iterrows():
                st.write(f"Email {index + 1} - Label: {row['Label']}")

            # Save the DataFrame with the new 'Label' column
            df.to_csv('result_dataset.csv', index=False)

            # Download link for the resulting dataset with labels
            st.markdown("### Download Resulting Dataset with Labels")
            st.write("Click below to download the CSV file.")
            st.markdown(get_binary_file_downloader_html('result_dataset.csv', 'Result Dataset'), unsafe_allow_html=True)

            # Add spam vs. ham visualization
            st.subheader("Spam vs. Ham Visualization:")
            data = df[['Label', 'Text']]
            data['length'] = data['Text'].apply(len)
            fig, ax = plt.subplots(figsize=(12, 4))
            data.hist(column='length', by='Label', bins=5, ax=ax)
            st.pyplot(fig)

        else:
            st.warning("Please enter your Gmail address and password.")

# Function to create download link for a file
def get_binary_file_downloader_html(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{file_label}.csv">Download {file_label}</a>'

if __name__ == "__main__":
    main()
