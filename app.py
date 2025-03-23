from flask import Flask, render_template, request, send_file
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import os
import csv
from io import StringIO
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Priority, category, tone, and spam keywords
high_priority_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
medium_priority_keywords = ['soon', 'please', 'today', 'quickly']
low_priority_keywords = ['whenever', 'no rush', 'later']
category_keywords = {
    'Technical': ['server', 'fix', 'issue', 'error', 'bug', 'crash', 'system'],
    'Meeting': ['discuss', 'meeting', 'timeline', 'availability', 'schedule'],
    'Feedback': ['feedback', 'suggestion', 'update', 'review', 'opinion'],
    'Sales': ['order', 'purchase', 'invoice', 'payment', 'deal'],
    'Support': ['help', 'support', 'problem', 'assistance', 'query']
}
tone_keywords = {
    'Formal': ['regards', 'sincerely', 'respectfully', 'please', 'thank you'],
    'Casual': ['hey', 'hi', 'cool', 'chill', 'buddy'],
    'Angry': ['unacceptable', 'angry', 'furious', 'disappointing', 'outrage']
}
spam_keywords = ['win', 'free', 'prize', 'click here', 'urgent money', 'verify your account']

# Predefined emails with longer bodies
predefined_emails = [
    {"subject": "Urgent: Server Down", "body": "Our main server has been down since 9 AM this morning. This is a critical issue affecting all users. Please fix this immediately to restore service. Contact me if you need more details or access logs. Time is of the essence!"},
    {"subject": "Meeting Today", "body": "Hi team, can we discuss the project timeline soon? I’d like to go over the milestones and resource allocation for the next sprint. Please let me know your availability today so we can set something up quickly. Thanks for your cooperation!"},
    {"subject": "No Rush", "body": "Hello, I just wanted to share some feedback on the recent update. It’s not urgent, so take your time replying to this. Whenever you get a chance, let me know how we can incorporate these suggestions into the next release. No pressure at all!"}
]

# Feedback storage
FEEDBACK_FILE = 'feedback.json'
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump({}, f)

# Function to analyze email
def analyze_email(subject, body, language='en'):
    full_text = subject + " " + body
    full_text_lower = full_text.lower()
    sentiment_scores = sid.polarity_scores(full_text)
    compound_score = sentiment_scores['compound']
    pos_score = sentiment_scores['pos'] * 100
    neg_score = sentiment_scores['neg'] * 100
    neu_score = sentiment_scores['neu'] * 100

    # Sentiment
    sentiment = "Positive" if compound_score >= 0.05 else "Negative" if compound_score <= -0.05 else "Neutral"

    # Priority
    triggered_keywords = []
    if any(kw in full_text_lower for kw in high_priority_keywords):
        priority = "High"
        triggered_keywords = [kw for kw in high_priority_keywords if kw in full_text_lower]
    elif any(kw in full_text_lower for kw in medium_priority_keywords):
        priority = "Medium"
        triggered_keywords = [kw for kw in medium_priority_keywords if kw in full_text_lower]
    else:
        priority = "Low"
        triggered_keywords = [kw for kw in low_priority_keywords if kw in full_text_lower] or ["none"]

    # Categories (multi-category)
    categories = [cat for cat, kws in category_keywords.items() if any(kw in full_text_lower for kw in kws)]

    # Tone
    tones = [tone for tone, kws in tone_keywords.items() if any(kw in full_text_lower for kw in kws)]
    tone = tones[0] if tones else "Neutral"

    # Spam/Phishing Detection
    is_spam = any(kw in full_text_lower for kw in spam_keywords)

    # Summarization
    sentences = sent_tokenize(full_text)
    summary = " ".join(sentences[:2]) if len(sentences) > 2 else full_text

    # Actionable suggestion and response time
    if is_spam:
        suggestion = "Flag as potential spam/phishing and avoid responding."
        response_time = "N/A"
    elif priority == "High" and sentiment == "Negative":
        suggestion = "Escalate to IT/supervisor immediately."
        response_time = "Within 1 hour"
    elif priority == "High":
        suggestion = "Address promptly with relevant team."
        response_time = "Within 4 hours"
    elif priority == "Medium" and sentiment == "Negative":
        suggestion = "Review and respond with clarification."
        response_time = "Within 24 hours"
    elif priority == "Medium":
        suggestion = "Schedule a follow-up if needed."
        response_time = "Within 48 hours"
    else:
        suggestion = "Reply at your convenience."
        response_time = "Within 1 week"

    # Automated reply suggestion
    if is_spam:
        reply_suggestion = "N/A"
    elif sentiment == "Negative":
        reply_suggestion = f"Subject: Re: {subject}\n\nDear sender, I’m sorry to hear about this issue. {suggestion} I’ll ensure it’s addressed promptly. Please let me know if you need further assistance."
    else:
        reply_suggestion = f"Subject: Re: {subject}\n\nHi, thanks for your email! {suggestion} I’ll get back to you soon with more details."

    return {
        'subject': subject, 'body': body, 'sentiment': sentiment, 'priority': priority, 'keywords': triggered_keywords,
        'pos_score': pos_score, 'neg_score': neg_score, 'neu_score': neu_score, 'categories': categories or ["Unknown"],
        'tone': tone, 'is_spam': is_spam, 'summary': summary, 'suggestion': suggestion, 'response_time': response_time,
        'reply_suggestion': reply_suggestion, 'language': language
    }

# Function to process emails from a file
def process_file(file_path):
    results = []
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            subject = lines[i].strip()
            body = lines[i + 1].strip() if i + 1 < len(lines) else ""
            results.append(analyze_email(subject, body))
    return sorted(results, key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}[x['priority']])  # Sort by priority

# Function to save results to text
def save_results_txt(results):
    with open('results.txt', 'w', encoding='utf-8') as file:
        for i, res in enumerate(results, 1):
            file.write(f"Email {i}: {res['subject']}\n{res['body']}\nSentiment: {res['sentiment']} (Pos: {res['pos_score']:.1f}%, Neg: {res['neg_score']:.1f}%, Neu: {res['neu_score']:.1f}%)\nPriority: {res['priority']} (Keywords: {', '.join(res['keywords'])})\nCategories: {', '.join(res['categories'])}\nTone: {res['tone']}\nSpam: {res['is_spam']}\nSummary: {res['summary']}\nSuggestion: {res['suggestion']}\nResponse Time: {res['response_time']}\nReply Suggestion: {res['reply_suggestion']}\n------------------------\n")
    return 'results.txt'

# Function to save results to CSV
def save_results_csv(results):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=['subject', 'body', 'sentiment', 'priority', 'keywords', 'pos_score', 'neg_score', 'neu_score', 'categories', 'tone', 'is_spam', 'summary', 'suggestion', 'response_time', 'reply_suggestion'])
    writer.writeheader()
    for res in results:
        res['keywords'] = ', '.join(res['keywords'])
        res['categories'] = ', '.join(res['categories'])
        writer.writerow(res)
    output.seek(0)
    return output

# Function to save results to PDF
def save_results_pdf(results):
    pdf_file = 'results.pdf'
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for i, res in enumerate(results, 1):
        story.append(Paragraph(f"Email {i}: {res['subject']}", styles['Heading2']))
        story.append(Paragraph(f"Body: {res['body']}", styles['BodyText']))
        story.append(Paragraph(f"Sentiment: {res['sentiment']} (Pos: {res['pos_score']:.1f}%, Neg: {res['neg_score']:.1f}%, Neu: {res['neu_score']:.1f}%)", styles['BodyText']))
        story.append(Paragraph(f"Priority: {res['priority']} (Keywords: {', '.join(res['keywords'])})", styles['BodyText']))
        story.append(Paragraph(f"Categories: {', '.join(res['categories'])}", styles['BodyText']))
        story.append(Paragraph(f"Tone: {res['tone']}", styles['BodyText']))
        story.append(Paragraph(f"Spam: {res['is_spam']}", styles['BodyText']))
        story.append(Paragraph(f"Summary: {res['summary']}", styles['BodyText']))
        story.append(Paragraph(f"Suggestion: {res['suggestion']}", styles['BodyText']))
        story.append(Paragraph(f"Response Time: {res['response_time']}", styles['BodyText']))
        story.append(Paragraph(f"Reply Suggestion: {res['reply_suggestion']}", styles['BodyText']))
        story.append(Spacer(1, 12))
    doc.build(story)
    return pdf_file

# Function to calculate analytics
def get_analytics(results):
    total = len(results)
    if total == 0:
        return {}
    high = sum(1 for r in results if r['priority'] == 'High') / total * 100
    medium = sum(1 for r in results if r['priority'] == 'Medium') / total * 100
    low = sum(1 for r in results if r['priority'] == 'Low') / total * 100
    spam = sum(1 for r in results if r['is_spam']) / total * 100
    return {'high': high, 'medium': medium, 'low': low, 'spam': spam, 'total': total}

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    file_saved = None
    analytics = None
    search_query = request.args.get('search', '')

    if request.method == 'POST':
        if 'email_select' in request.form:
            email_index = int(request.form['email_select'])
            selected_email = predefined_emails[email_index]
            results = [analyze_email(selected_email['subject'], selected_email['body'])]
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                error = "No file selected."
            elif not file.filename.endswith('.txt'):
                error = "Please upload a .txt file."
            else:
                file_path = os.path.join('uploads', file.filename)
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                file.save(file_path)
                results = process_file(file_path)
                if results is None:
                    error = f"Error processing file '{file.filename}'."
                elif 'save' in request.form and request.form['save'] == 'yes':
                    file_saved = save_results_txt(results)
                analytics = get_analytics(results)

        # Feedback submission
        if 'feedback' in request.form:
            email_id = request.form['email_id']
            feedback = request.form['feedback']
            with open(FEEDBACK_FILE, 'r+') as f:
                data = json.load(f)
                data[email_id] = feedback
                f.seek(0)
                json.dump(data, f)

        # Export options
        if 'export_csv' in request.form and results:
            return send_file(save_results_csv(results), mimetype='text/csv', as_attachment=True, download_name='email_analysis_results.csv')
        if 'export_pdf' in request.form and results:
            return send_file(save_results_pdf(results), as_attachment=True, download_name='email_analysis_results.pdf')

    # Filter results by search query
    if results and search_query:
        results = [r for r in results if search_query.lower() in r['subject'].lower() or search_query.lower() in r['body'].lower()]

    return render_template('index.html', results=results, error=error, file_saved=file_saved, predefined_emails=predefined_emails, analytics=analytics, search_query=search_query)

if __name__ == '__main__':
    app.run(debug=True)