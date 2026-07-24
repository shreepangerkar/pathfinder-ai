import os
import json
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load Environment Configuration
load_dotenv()

app = Flask(__name__, template_folder='../templates')

# Initialize Groq Client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

@app.route('/')
def serve_dashboard():
    """Renders the main single-page career planner."""
    return render_template(
        'index.html',
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", "")
    )

@app.route('/api/navigate', methods=['POST'])
def generate_strategy_matrix():
    """Generates 3 practical career path options using AI."""
    try:
        data = request.get_json() or {}
        
        academic_baseline = data.get('academicBaseline', 'Not Specified')
        work_style = data.get('workStyle', 'Not Specified')
        selected_skills = data.get('selectedSkills', [])
        geolocation = data.get('geolocation', 'Not Specified')
        custom_path = data.get('customPath', '')

        skills_formatted = ", ".join(selected_skills) if selected_skills else "General Capabilities"

        # Prompt instructing the AI to use plain, friendly, and human English
        system_instruction = """You are PathFinder AI, a friendly and practical career guidance mentor. 
Your goal is to give clear, encouraging, and easy-to-understand career roadmaps. Avoid unnecessary jargon, sci-fi buzzwords, or robotic phrases.

Return ONLY a raw, well-formed JSON object matching this exact structure:
{
  "careers": [
    {
      "title": "Exact Role Title",
      "alignment_type": "Best Fit",
      "description": "A clear, simple 2-sentence summary of what this role is and why it fits.",
      "future_proof_index": 85,
      "automation_risk": "Low",
      "market_trend_title": "Job Market Demand",
      "market_trend_description": "A brief, realistic note on hiring trends and industry demand.",
      "certifications": ["Certification 1", "Certification 2"],
      "timeline": {
        "year_1": {"title": "Entry Role Title", "salary": "$85,000", "percentage": 40},
        "year_5": {"title": "Mid-Level Role Title", "salary": "$135,000", "percentage": 72},
        "year_10": {"title": "Senior / Lead Title", "salary": "$210,000", "percentage": 100}
      },
      "day_in_life": [
        {"time": "09:00", "task": "Simple description of a typical morning task."},
        {"time": "13:00", "task": "Simple description of an afternoon task."}
      ],
      "skill_gap_steps": [
        {"step": "STEP 01", "description": "Practical first step to learn core basics."},
        {"step": "STEP 02", "description": "Build a hands-on project or portfolio piece."},
        {"step": "STEP 03", "description": "Apply for entry roles or internships."}
      ]
    }
  ]
}

Provide 3 distinct career options. The 'alignment_type' MUST strictly be either 'Best Fit', 'High Growth', or 'Alternative Path'. The 'automation_risk' MUST strictly be 'Low', 'Moderate', or 'High'."""

        user_context_prompt = (
            f"Please generate 3 tailored career paths for a candidate with the following profile:\n"
            f"- Education / Stage: {academic_baseline}\n"
            f"- Preferred Work Style: {work_style}\n"
            f"- Top Skills: {skills_formatted}\n"
            f"- Preferred Location: {geolocation}\n"
            f"- Specific Interest / Target Role: {custom_path if custom_path else 'None'}\n"
        )

        # Call Groq active model Llama 3.3 70B
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_context_prompt}
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        raw_response = completion.choices[0].message.content
        structured_json = json.loads(raw_response)
        
        return jsonify(structured_json)

    except Exception as e:
        return jsonify({"error": f"Unable to generate recommendations: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
