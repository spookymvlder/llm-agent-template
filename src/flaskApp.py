from flask import Flask, flash, redirect, render_template, request, session, send_file, jsonify
from flask_session import Session

import os
import logging
from setupAgent import agent, logger

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

Session(app)

@app.route("/", methods=["GET", "POST"])
def index():
    """Main chat interface"""
    if request.method == "POST":
        user_message = request.form.get("message", "").strip()
        
        if not user_message:
            flash("Please enter a message", "error")
            return render_template("index.html")
        
        try:
            # Get agent response
            response = agent.chat(user_message)
            
            # Store conversation in session
            if "conversation" not in session:
                session["conversation"] = []
            
            session["conversation"].append({
                "user": user_message,
                "agent": str(response),
                "timestamp": os.time()
            })
            
            return render_template("index.html", 
                                 conversation=session.get("conversation", []),
                                 latest_response=str(response))
        
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            flash(f"Error: {str(e)}", "error")
            return render_template("index.html")
    
    return render_template("index.html", conversation=session.get("conversation", []))

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """REST API endpoint for chat"""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        response = agent.chat(message)
        
        return jsonify({
            "response": str(response),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"API chat error: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/api/analyze_url", methods=["POST"])
def api_analyze_url():
    """API endpoint for URL analysis"""
    try:
        data = request.get_json()
        url = data.get("url", "").strip()
        time_limit = data.get("time_limit_days")
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Use the agent's URL analysis tool
        if time_limit:
            prompt = f"Using the media analyzer tool, extract themes from this content: {url}. Focus on the last {time_limit} days."
        else:
            prompt = f"Using the media analyzer tool, extract themes from this content: {url}."
        
        response = agent.chat(prompt)
        
        return jsonify({
            "url": url,
            "analysis": str(response),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"URL analysis error: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/clear", methods=["POST"])
def clear_conversation():
    """Clear conversation history"""
    session.pop("conversation", None)
    flash("Conversation cleared", "info")
    return redirect("/")

@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Quick agent test
        test_response = agent.chat("Hello")
        return jsonify({
            "status": "healthy",
            "agent_responsive": True,
            "message": "Agent is working properly"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy", 
            "agent_responsive": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)