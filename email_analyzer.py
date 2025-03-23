# Import required libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os

# Download VADER lexicon (run only once)
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Priority keywords
high_priority_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
medium_priority_keywords = ['soon', 'please', 'today', 'quickly']
low_priority_keywords = ['whenever', 'no rush', 'later']

# Function to analyze email
def analyze_email(subject, body):
    # Combine subject and body for full analysis
    full_text = subject + " " + body

    # Sentiment analysis
    sentiment_scores = sid.polarity_scores(full_text)
    compound_score = sentiment_scores['compound']
    pos_score = sentiment_scores['pos'] * 100
    neg_score = sentiment_scores['neg'] * 100
    neu_score = sentiment_scores['neu'] * 100

    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Priority analysis
    full_text_lower = full_text.lower()
    triggered_keywords = []

    if any(keyword in full_text_lower for keyword in high_priority_keywords):
        priority = "High"
        triggered_keywords = [kw for kw in high_priority_keywords if kw in full_text_lower]
    elif any(keyword in full_text_lower for keyword in medium_priority_keywords):
        priority = "Medium"
        triggered_keywords = [kw for kw in medium_priority_keywords if kw in full_text_lower]
    else:
        priority = "Low"
        triggered_keywords = [kw for kw in low_priority_keywords if kw in full_text_lower] or ["none"]

    return sentiment, priority, triggered_keywords, pos_score, neg_score, neu_score

# Function to process emails from a file
def process_file(filename):
    results = []
    if not os.path.exists(filename):
        return None
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):  # Assuming subject on odd lines, body on even
            subject = lines[i].strip()
            body = lines[i + 1].strip() if i + 1 < len(lines) else ""
            sentiment, priority, keywords, pos, neg, neu = analyze_email(subject, body)
            results.append({
                'subject': subject,
                'body': body,
                'sentiment': sentiment,
                'priority': priority,
                'keywords': keywords,
                'pos_score': pos,
                'neg_score': neg,
                'neu_score': neu
            })
    return results

# Function to save results to a file
def save_results(results, output_filename="results.txt"):
    with open(output_filename, 'w', encoding='utf-8') as file:
        for i, res in enumerate(results, 1):
            file.write(f"Email {i}:\n")
            file.write(f"Subject: {res['subject']}\n")
            file.write(f"Body: {res['body']}\n")
            file.write(f"Sentiment: {res['sentiment']} (Pos: {res['pos_score']:.1f}%, Neg: {res['neg_score']:.1f}%, Neu: {res['neu_score']:.1f}%)\n")
            file.write(f"Priority: {res['priority']} (Triggered Keywords: {', '.join(res['keywords'])})\n")
            file.write("------------------------\n")

# Main program
def main():
    print("Email Sentiment and Priority Analyzer")
    print("------------------------------------")
    choice = input("Enter '1' to analyze a single email or '2' to process a file: ")

    if choice == '1':
        subject = input("Enter the email subject: ")
        body = input("Enter the email body: ")
        sentiment, priority, keywords, pos, neg, neu = analyze_email(subject, body)
        print("\nResults:")
        print(f"Sentiment: {sentiment} (Pos: {pos:.1f}%, Neg: {neg:.1f}%, Neu: {neu:.1f}%)")
        print(f"Priority: {priority} (Triggered Keywords: {', '.join(keywords)})")
    
    elif choice == '2':
        filename = input("Enter the filename (e.g., emails.txt): ")
        results = process_file(filename)
        if results is None:
            print(f"Error: File '{filename}' not found.")
        else:
            for i, res in enumerate(results, 1):
                print(f"\nEmail {i}:")
                print(f"Subject: {res['subject']}")
                print(f"Body: {res['body']}")
                print(f"Sentiment: {res['sentiment']} (Pos: {res['pos_score']:.1f}%, Neg: {res['neg_score']:.1f}%, Neu: {res['neu_score']:.1f}%)")
                print(f"Priority: {res['priority']} (Triggered Keywords: {', '.join(res['keywords'])})")

            save_choice = input("Save results to 'results.txt'? (y/n): ").lower()
            if save_choice == 'y':
                save_results(results)
                print("Results saved to 'results.txt'.")
    
    else:
        print("Invalid choice. Please enter '1' or '2'.")

# Run the program
if __name__ == "__2main__":
    main()