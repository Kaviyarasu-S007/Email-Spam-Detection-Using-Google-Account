```markdown
# Email Classifier App

This Streamlit app reads emails from a Gmail account, performs text preprocessing and topic modeling using LDA, classifies emails as spam or ham, and visualizes spam vs. ham distribution.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Kaviyarasu-S007/Email-Spam-Detection-Using-Google-Account.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
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
