import os
import json
from flask import Flask, render_template, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Initialize Environment Configuration
load_dotenv()

app = Flask(__name__, template_folder='../templates')

# Initialize the Groq Cloud Service Engine client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

@app.route('/')
def serve_dashboard():
    """Renders the foundational single-page glassmorphic user workspace."""
    return render_template('index.html')

@app.route('/api/navigate', methods=['POST'])
def generate_strategy_matrix():
    """
    Ingests client criteria payload, initializes predictive modeling parameters,
    and returns a structured JSON evaluation framework from the Groq active engine.
    """
    try:
        data = request.get_json() or {}
        
        academic_baseline = data.get('academicBaseline', 'Not Specified')
        work_style = data.get('workStyle', 'Not Specified')
        selected_skills = data.get('selectedSkills', [])
        geolocation = data.get('geolocation', 'Not Specified')
        custom_path = data.get('customPath', '')

        skills_formatted = ", ".join(selected_skills) if selected_skills else "General Capabilities"

        # System Directive specifying exact structural integrity mapping for UI data layer parsing
        system_instruction = """You are the PathFinder AI Analytical Matrix Engine. You output strictly raw, well-formed JSON objects. 
Do not include markdown wrappers, thoughts, trailing text, or code block markers. 
Your response must rigidly match this structural layout schema:
{
  "careers": [
    {
      "title": "Exact Professional Title",
      "alignment_type": "Optimal Alignment",
      "description": "High-level overview description.",
      "future_proof_index": 85,
      "automation_risk": "Low",
      "market_trend_title": "Labor Trend Title Header",
      "market_trend_description": "Detailed multi-sentence labor movement analysis.",
      "certifications": ["Cert 1", "Cert 2"],
      "timeline": {
        "year_1": {"title": "Entry Role Title", "salary": "$85,000", "percentage": 40},
        "year_5": {"title": "Senior Role Title", "salary": "$135,000", "percentage": 72},
        "year_10": {"title": "Principal/Director Title", "salary": "$210,000", "percentage": 100}
      },
      "day_in_life": [
        {"time": "09:00", "task": "Core system review protocol description."},
        {"time": "13:00", "task": "Collaborative sprint architecture task."}
      ],
      "skill_gap_steps": [
        {"step": "STEP 01", "description": "Acquire baseline platform credential details."},
        {"step": "STEP 02", "description": "Construct operational portfolios within domain spaces."},
        {"step": "STEP 03", "description": "Target high-density entry hiring tracks."}
      ]
    }
  ]
}
Generate 3 highly tailored career tracks using the provided telemetry framework. The alignment_type must strictly be either 'Optimal Alignment', 'High Growth', or 'Strategic Pivot'. The automation_risk must strictly be either 'Low', 'Moderate', or 'High'."""

        user_context_prompt = (
            f"Compute career matrix parameters matching the following client tracking variables:\n"
            f"- Academic Standing Baseline: {academic_baseline}\n"
            f"- Work-Style Core DNA: {work_style}\n"
            f"- Selected Technical Core Competencies: {skills_formatted}\n"
            f"- Targeted Geo Labor Market: {geolocation}\n"
            f"- Special Track Filter: {custom_path if custom_path else 'None Specified'}\n\n"
            f"Analyze automation indexing vectors relative to current operational advancements."
        )

        # Execute ultra-high-speed LLM completion via active Llama 3.3 70B model on Groq
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
        return jsonify({"error": f"Internal execution anomaly detected: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)