# medical-chatbot-with-fine-tune-LLM
This project focuses on fine-tuning Meta’s LLaMA 2 model to develop a domain-specific medical chatbot capable of understanding and responding to patient and clinician queries with high accuracy. Leveraging parameter-efficient fine-tuning techniques—LoRA and QLoRA the project ensures resource-efficient training while maintaining high performance.
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Medical Chatbot using Fine-Tuned LLaMA 2</title>
</head>
<body>

<h1>🩺 Medical Chatbot using Fine-Tuned LLaMA 2 with LoRA/QLoRA</h1>

<p>
  This project fine-tunes Meta's <a href="https://ai.meta.com/llama/" target="_blank">LLaMA 2</a> model using LoRA and QLoRA techniques to build a domain-specific <strong>medical chatbot</strong>. 
  The chatbot is trained on curated medical text data and is capable of answering patient and clinician queries with context awareness and medical relevance.
</p>

<hr>

<h2>🚀 Project Features</h2>
<ul>
  <li>🔬 Fine-tuning LLaMA 2 with <strong>LoRA</strong> and <strong>QLoRA</strong> for efficient training</li>
  <li>🧹 Preprocessing and structuring of medical datasets</li>
  <li>🏥 Domain-specific prompt-response dataset formatting</li>
  <li>💾 Saving and loading fine-tuned model and tokenizer</li>
  <li>🧠 Knowledge-aware, safe, and medically coherent chatbot</li>
</ul>

<hr>

<h2>📂 Directory Structure</h2>
<pre><code>
medical-llama-chatbot/
│
├── data/                 # Raw and preprocessed medical text data
├── scripts/              # Training and preprocessing scripts
  
├── notebooks/            # Experiment notebooks
├── config/               # Model and training config files
└── README.md             # Project documentation
</code></pre>

<hr>

<h2>⚙️ Setup Instructions</h2>

<h3>1. Clone the repository</h3>
<pre><code>git clone https://github.com/yourusername/medical-llama-chatbot.git
cd medical-llama-chatbot
</code></pre>

<h3>2. Install Dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<p><strong>💡 Recommended:</strong> Use a virtual environment or Docker for reproducibility.</p>

<hr>

<h2>🧪 Fine-Tuning Process</h2>

<ol>
  <li><strong>Data Preprocessing and CSV File Structure</strong><br>
      Clean, tokenize, and format the medical text into instruction-style prompts.<br>
      The training csv file should be in the below formate<br>
          <pre><code> 
      {'Column_name': '<s>[INST]Q. Can kitten allergy in family members make me allergic too?[/INST] Hi.  Allergy can run in families, you have a chance of being allergic as your father and brother are allergic to cats but not a must. Being close to cats in the past for many years is a good indicator that you might not be allergic to cats but still allergy can develop at any age. You can do a skin allergy test by a dermatologist (skin doctor) to know if you are allergic to cats or not. Skin allergy test. ?Allergy. Avoid contact with cats, if proved to be allergic.</s>'}
      </code></pre>
    
  </li>
  <li><strong>Model Training with LoRA/QLoRA</strong><br>
      <pre><code>python scripts/train_model.py </code></pre>
  </li>
  <li><strong>Chatting with Bot</strong><br>
      <pre><code>
 <pre><code>python scripts/inferrence.py </code></pre>
      </code></pre>
  </li>
</ol>

<hr>

<h2>💬 Inference Example</h2>

<pre><code>from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer/")
model = AutoModelForCausalLM.from_pretrained("models/llama-med/")

prompt = "Patient reports chest pain and shortness of breath. What could be the cause?"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
</code></pre>

<hr>

<h2>📌 Requirements</h2>
<ul>
  <li>Python 3.10+</li>
  <li>Hugging Face Transformers</li>
  <li>PEFT / LoRA libraries</li>
  <li>BitsAndBytes (for QLoRA)</li>
</ul>

<hr>

<h2>📄 License</h2>
<p>
  This project is licensed under the MIT License. Note that medical data used in training is anonymized and derived from public or synthetic datasets.
</p>

<hr>

<h2>🤝 Contributions</h2>
<p>
  Pull requests and issues are welcome! Please ensure any medical contributions follow ethical AI and data privacy standards.
</p>

<hr>

<h2>📬 Contact</h2>
<p>
  For questions, reach out via <a href="https://github.com/medical-chatbot-with-fine-tune-LLM/issues">GitHub Issues</a> or email 
  <a href="mailto:mohsinraza2999@gmail.com">mohsinraza2999@gmail.com</a>.
</p>
<h2>📬 Auther</h2>
<p>Mohsin Raza</p>
</body>
</html>
