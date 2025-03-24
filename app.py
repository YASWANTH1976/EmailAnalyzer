from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from datetime import datetime

# Download NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'  # Enable session support

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Analysis Model (for storing user-specific analysis history)
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(200))
    sentiment = db.Column(db.String(50))
    priority = db.Column(db.String(50))
    tone = db.Column(db.String(50))
    categories = db.Column(db.String(200))
    spam_phishing = db.Column(db.String(50))
    summary = db.Column(db.Text)
    reply_suggestion = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('analyses', lazy=True))

# Email Model (for predefined emails)
class Email:
    def __init__(self, id, subject, body):
        self.id = id
        self.subject = subject
        self.body = body

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load predefined emails from emails.txt
def load_emails():
    emails = []
    try:
        with open('emails.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                subject = lines[i].strip()
                body = lines[i+1].strip()
                emails.append(Email(i//2 + 1, subject, body))
    except FileNotFoundError:
        print("emails.txt not found. Please create the file with email data.")
    return emails

# Analyze email (with improved priority detection)
def analyze_email(subject, body):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(body)
    sentiment = 'positive' if scores['compound'] > 0.05 else 'negative' if scores['compound'] < -0.05 else 'neutral'

    # Combine subject and body for keyword checks
    combined_text = (subject + " " + body).lower()

    # Determine priority with expanded keywords
    high_priority_keywords = ['urgent', 'asap', 'immediately', 'critical']
    medium_priority_keywords = ['important', 'please', 'soon']
    low_priority_keywords = ['no rush', 'whenever', 'at your convenience']

    if any(kw in combined_text for kw in low_priority_keywords):
        priority = 'low'  # Low priority overrides others
    elif any(kw in combined_text for kw in high_priority_keywords):
        priority = 'high'
    elif any(kw in combined_text for kw in medium_priority_keywords):
        priority = 'medium'
    else:
        priority = 'low'

    tone = 'formal' if 'dear' in combined_text else 'casual'
    categories = ['Technical'] if 'error' in combined_text else ['General']
    spam_phishing = 'Yes' if 'win a prize' in combined_text else 'No'
    summary = body[:50] + '...' if len(body) > 50 else body
    reply_suggestion = 'Thank you for your email.' if sentiment == 'positive' else 'I’m sorry to hear about your issue.'
    return {
        'subject': subject,
        'sentiment': sentiment,
        'priority': priority,
        'tone': tone,
        'categories': categories,
        'spam_phishing': spam_phishing,
        'summary': summary,
        'reply_suggestion': reply_suggestion
    }

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    # Clear session data on logout
    session.pop('analysis_results', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    emails = load_emails()
    analysis_results = session.get('analysis_results', [])
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    analytics = {
        'total_emails': 0,
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'high_priority': 0,
        'medium_priority': 0,
        'low_priority': 0
    }

    # Handle search and sort (GET request)
    search_query = request.args.get('search', '').lower()
    sort_by = request.args.get('sort', 'priority')

    # Filter results based on search query
    if search_query and analysis_results:
        analysis_results = [
            result for result in analysis_results
            if search_query in result['subject'].lower() or search_query in result['summary'].lower()
        ]

    # Sort results
    if sort_by == 'priority':
        analysis_results.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x['priority'], 3))
    elif sort_by == 'sentiment':
        analysis_results.sort(key=lambda x: x['sentiment'])
    elif sort_by == 'tone':
        analysis_results.sort(key=lambda x: x['tone'])

    # Handle email analysis (POST request)
    if request.method == 'POST':
        if 'email_select' in request.form:
            selected_ids = request.form.getlist('email_select')
            selected_emails = [e for e in emails if str(e.id) in selected_ids]
            analysis_results = [analyze_email(email.subject, email.body) for email in selected_emails]
        elif 'email_file' in request.files:
            file = request.files['email_file']
            if file and file.filename.endswith('.txt'):
                content = file.read().decode('utf-8')
                lines = content.split('\n')
                subject = lines[0].strip()
                body = '\n'.join(lines[1:]).strip()
                analysis_results = [analyze_email(subject, body)]

        if analysis_results:
            # Save analysis results to database
            for result in analysis_results:
                analysis = Analysis(
                    user_id=current_user.id,
                    subject=result['subject'],
                    sentiment=result['sentiment'],
                    priority=result['priority'],
                    tone=result['tone'],
                    categories=','.join(result['categories']),
                    spam_phishing=result['spam_phishing'],
                    summary=result['summary'],
                    reply_suggestion=result['reply_suggestion']
                )
                db.session.add(analysis)
            db.session.commit()

            # Update analytics
            analytics['total_emails'] = len(analysis_results)
            for result in analysis_results:
                if result['sentiment'] == 'positive':
                    analytics['positive'] += 1
                elif result['sentiment'] == 'negative':
                    analytics['negative'] += 1
                else:
                    analytics['neutral'] += 1
                if result['priority'] == 'high':
                    analytics['high_priority'] += 1
                elif result['priority'] == 'medium':
                    analytics['medium_priority'] += 1
                else:
                    analytics['low_priority'] += 1
            # Store analysis results in session
            session['analysis_results'] = analysis_results

    return render_template('index.html', emails=emails, analysis_results=analysis_results, analytics=analytics, analyses=analyses)

@app.route('/export', methods=['POST'])
@login_required
def export():
    analysis_results = session.get('analysis_results', [])
    if not analysis_results:
        flash('No analysis results to export.', 'error')
        return redirect(url_for('index'))

    export_type = request.form['export_type']
    
    if export_type == 'csv':
        output = io.StringIO()
        output.write('Subject,Sentiment,Priority,Tone,Categories,Spam/Phishing,Summary,Reply Suggestion\n')
        for result in analysis_results:
            output.write(f"{result['subject']},{result['sentiment']},{result['priority']},{result['tone']},{','.join(result['categories'])},{result['spam_phishing']},{result['summary']},{result['reply_suggestion']}\n")
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='analysis_results.csv')
    
    elif export_type == 'pdf':
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        y = 750
        p.drawString(100, y, "Analysis Results")
        y -= 20
        for result in analysis_results:
            p.drawString(100, y, f"Subject: {result['subject']}")
            y -= 20
            p.drawString(100, y, f"Sentiment: {result['sentiment']}")
            y -= 20
            p.drawString(100, y, f"Priority: {result['priority']}")
            y -= 20
            if y < 50:
                p.showPage()
                y = 750
        p.showPage()
        p.save()
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name='analysis_results.pdf')
    
    elif export_type == 'txt':
        output = io.StringIO()
        for result in analysis_results:
            output.write(f"Subject: {result['subject']}\n")
            output.write(f"Sentiment: {result['sentiment']}\n")
            output.write(f"Priority: {result['priority']}\n")
            output.write(f"Tone: {result['tone']}\n")
            output.write(f"Categories: {','.join(result['categories'])}\n")
            output.write(f"Spam/Phishing: {result['spam_phishing']}\n")
            output.write(f"Summary: {result['summary']}\n")
            output.write(f"Reply Suggestion: {result['reply_suggestion']}\n\n")
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/plain', as_attachment=True, download_name='analysis_results.txt')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)