# Curistro - AI-Powered Learn-by-Teaching Platform

<p align="center">
  <strong>🎓 Teach a Curious AI Student to Master Any Concept</strong>
</p>

<p align="center">
  <em>An innovative educational tool based on the "Learning by Teaching" methodology, powered by OpenAI</em>
</p>

---

## 🌟 Overview

**Curistro** is an interactive web application that flips the traditional learning paradigm. Instead of being taught by an AI, *you become the teacher* and explain concepts to a curious AI student named "Curistro-Student". This approach leverages the proven pedagogical principle that teaching is one of the most effective ways to learn.

The AI student asks clarifying questions, challenges your understanding, and at the end of each session, generates a quiz to evaluate how well you've grasped the concept.

## ✨ Features

### 🗣️ Interactive Conversation
- **Real-time streaming responses** - Watch the AI student ask questions in real-time
- **Speech-to-Text** - Use your microphone to explain concepts verbally
- **Text-to-Speech** - Listen to AI responses for accessibility

### 📝 Smart Evaluation
- **Auto-generated Quiz** - 4 diagnostic questions covering:
  - Term Explanation
  - Component Analysis
  - Application
  - Concept Contrast
- **AI Self-Evaluation** - The AI student answers its own quiz to measure learning effectiveness

### 🔄 Adaptive Learning
- **Explain-Back Feature** - AI restates concepts in its own words to check understanding
- **Adaptive Recap** - Generates targeted micro-lessons for weak areas
- **Knowledge Map** - D3.js powered visualization of your knowledge graph

### 📤 Export Options
- Download session summaries as **Markdown** or **PDF**
- Full conversation history included

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible endpoint)
- ffmpeg (for audio transcription)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/curistro.git
   cd curistro
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask python-dotenv openai markdown fpdf pydub eventlet
   # Optional for PDF with WeasyPrint:
   pip install weasyprint
   ```

4. **Configure API key**
   
   Create an `api.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser** and navigate to `http://localhost:5000`

## 📖 How to Use

### Step 1: Start a Teaching Session
- Enter a **Topic Title** (e.g., "Binary Search Trees")
- Write your **Initial Explanation** of the concept
- Add a **Category** (e.g., "Data Structures")
- Optionally add **Related Topics** for the knowledge graph

### Step 2: Teach the AI
- The AI student will ask clarifying questions
- Respond with detailed explanations
- Use the **Explain-Back** button to check how well the AI understands
- Use 🎤 to speak and 🔊 to hear responses

### Step 3: Evaluate & Review
- Click **Finish & Evaluate** to generate a quiz
- Review the AI's answers and your teaching effectiveness
- Generate an **Adaptive Recap** for topics that need reinforcement
- View your **Knowledge Map** showing concept relationships
- **Export** your session as Markdown or PDF

## 🏗️ Project Structure

```
curistro/
├── app.py              # Main Flask application
├── api.env             # API configuration (create this)
├── static/
│   ├── style.css       # Styling
│   └── fonts/          # DejaVu fonts for PDF export
└── templates/
    ├── index.html      # Home page - start teaching
    ├── conversation.html # Chat interface
    └── result.html     # Quiz results & evaluation
```

## 🔧 Configuration

### Change AI Model
In `app.py`, modify the `StudentAI` class constants:
```python
CHAT_MODEL = "o4-mini"    # For conversation
QUIZ_MODEL = "o4-mini"    # For quiz generation
TEMPERATURE = 1.0
```

### Custom API Endpoint
Modify the base URL in `app.py`:
```python
client = OpenAI(api_key=api_key, base_url="https://your-api-endpoint/v1")
```

## 📚 The Learning Science Behind Curistro

Curistro is built on several evidence-based learning principles:

1. **Protégé Effect** - Students learn better when they teach others
2. **Active Recall** - Explaining forces you to retrieve information
3. **Elaborative Interrogation** - Answering "why" questions deepens understanding
4. **Diagnostic Assessment** - Multi-dimensional quizzes identify knowledge gaps
5. **Targeted Remediation** - Adaptive recaps focus on weak areas

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- OpenAI for the powerful language models
- D3.js for knowledge map visualization
- The "Learning by Teaching" research community

---

<p align="center">
  <strong>Learn better by teaching. Start your session today! 🚀</strong>
</p>
