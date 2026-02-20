import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

def analyze_sentiment(text, llm):
    reasoning_prompt = f"""
    You are a professional customer insights analyst.

    Your task is to classify the sentiment of the following review as
    Positive, Negative, or Neutral using the definitions below, and explain
    your reasoning step by step.

    IMPORTANT:
    - The reasoning MUST be written in Arabic only.
    - Do NOT use English words or phrases in the reasoning.

    Sentiment definitions:
    - Positive: Clear satisfaction, praise, enjoyment, or approval.
    - Negative: Clear dissatisfaction, criticism, dislike, or disappointment.
    - Neutral: Acceptable, indifferent, mixed, or weak emotion
    (e.g., "okay", "fine", "average", "nothing special").

    Instructions:
    - Identify key words, phrases, or expressions that indicate sentiment.
    - Decide the sentiment first, then justify it.

    Review: "{text}"

    Format your output exactly as:

    Sentiment: <Positive / Negative / Neutral>
    Reasoning: <step-by-step explanation in Arabic>
    """
    
    reliable_prompt = f"""
    You are a professional customer insights analyst.

    Your task is to classify the sentiment of the following review as
    Positive, Negative, or Neutral using the same definitions below.

    Sentiment definitions:
    - Positive: Clear satisfaction, praise, enjoyment, or approval.
    - Negative: Clear dissatisfaction, criticism, dislike, or disappointment.
    - Neutral: Acceptable, indifferent, mixed, or weak emotion
    (e.g., "okay", "fine", "average", "nothing special").

    Instructions:
    - Decide the single best sentiment.
    - If the emotion is mild or unenthusiastic, choose Neutral.
    - Output ONLY the sentiment label.
    - Do not explain.

    Review: "{text}"
    """
    
    reasoning_response = llm.invoke(reasoning_prompt, temperature=0.5)
    reliable_response = llm.invoke(reliable_prompt, temperature=0.0)
    
    return {
        "text": text,
        "reasoning_output": reasoning_response.content,
        "reliable_output": reliable_response.content
    }

texts = [
    "الموبايل ممتاز والبطارية بتقعد وقت طويل جدًا.",
    "الفيلم كان ممل وطويل زيادة عن اللزوم.",
    "عادي، مش وحش بس مش أحسن حاجة."
]

results = [analyze_sentiment(text, llm) for text in texts]

with open("sentiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Results saved to sentiment_results.json")