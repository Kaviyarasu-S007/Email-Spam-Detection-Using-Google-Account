# Email Classifier App

This Streamlit app reads emails from a Gmail account, performs text preprocessing and topic modeling using LDA, classifies emails as spam or ham, and visualizes spam vs. ham distribution.

# To extract emails, do followings
Extract selected mails from your gmail account

1. Make sure you enable IMAP in your gmail settings
(Log on to your Gmail account and go to Settings, See All Settings, and select
 Forwarding and POP/IMAP tab. In the "IMAP access" section, select Enable IMAP.)

2. If you have 2-factor authentication, gmail requires you to create an application
specific password that you need to use. 
Go to your Google account settings and click on 'Security'.
Scroll down to App Passwords under 2 step verification.
Select Mail under Select App. and Other under Select Device. (Give a name, e.g., python)
The system gives you a password that you need to use to authenticate from python.


## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Kaviyarasu-S007/Email-Spam-Detection-Using-Google-Account.git
   ```

2. Install the required dependencies:

   ```bash
   pip install imaplib
   pip install email
   pip install traceback
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Enter your Gmail address and password, and click "Start Email Classification" to analyze and classify emails.

## File Descriptions

- `app.py`: Main Streamlit application.
- `requirements.txt`: List of Python dependencies.
- `result_dataset.csv`: CSV file containing the resulting dataset with email labels.
- `emails_dataset.csv`: CSV file containing the original emails dataset.

## Usage

1. Enter your Gmail address and password.
2. Click "Start Email Classification" to process and classify emails.
3. View the resulting dataset, email topics, and labels in the Streamlit app.
4. Download the resulting dataset with labels.
