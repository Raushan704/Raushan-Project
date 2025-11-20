# 🧠 Quill bot

An AI-powered chatbot that processes PDF documents and answers questions about their content using natural language processing and machine learning techniques.

## 🚀 Features

- **PDF Text Extraction**: Automatically extracts and processes text from multiple PDF files
- **Intelligent Text Processing**: Advanced preprocessing with NLTK for optimal text analysis
- **Similarity-Based Search**: Uses TF-IDF vectorization and cosine similarity for accurate content matching
- **Interactive Web Interface**: Built with Streamlit for a user-friendly chat experience
- **Real-time Q&A**: Ask questions about your PDF content and get relevant answers instantly

## 📁 Project Structure

```
Lnagchain 1/
├── chatbot.py              # Main chatbot application
├── requirements.txt        # Python dependencies
├── pdf data/              # Directory containing PDF files
│   ├── Team Hope Presentation-final.pdf
│   └── Untitled document.pdf
├── .venv/                 # Virtual environment
└── README.md              # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.13** - Core programming language
- **Streamlit** - Web application framework for the chat interface
- **PyPDF2** - PDF text extraction library
- **NLTK** - Natural Language Toolkit for text preprocessing
- **scikit-learn** - Machine learning library for:
  - TF-IDF Vectorization
  - Cosine Similarity calculations
- **NumPy** - Numerical computing support
- **TensorFlow** - Deep learning framework (for future enhancements)

## 📋 Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- PDF files in the `pdf data/` directory

## ⚙️ Installation

1. **Clone or download the project**
   ```bash
   cd "/home/som/Desktop/Lnagchain 1"
   ```

2. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF files**
   - Place your PDF documents in the `pdf data/` directory
   - The chatbot will automatically process all `.pdf` files in this folder

## 🚀 Usage

### Running the Chatbot

1. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Start the Streamlit application**
   ```bash
   .venv/bin/python -m streamlit run chatbot.py
   ```

3. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:8501`
   - The chatbot interface will load automatically

### Using the Chatbot

1. **Ask Questions**: Type your questions about the PDF content in the text input field
2. **Get Answers**: Click the "Ask" button to receive relevant responses
3. **View Confidence**: Each response includes a confidence score indicating relevance

### Example Queries

- "What is a biological neuron?"
- "Explain the structure of neural networks"
- "What are the main components discussed?"
- "Tell me about the project presentation"

## 🔧 How It Works

### 1. PDF Processing
- Extracts text from all PDF files in the `pdf data/` directory
- Handles multi-page documents automatically
- Processes over 50,000+ characters of content

### 2. Text Preprocessing
- Removes special characters and unnecessary symbols
- Converts text to lowercase for consistency
- Filters out stop words using NLTK
- Splits content into meaningful sentences/chunks

### 3. Vectorization
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Creates a vocabulary of up to 5,000 features
- Converts text chunks into numerical vectors for similarity calculations

### 4. Query Processing
- Transforms user questions into the same vector space
- Calculates cosine similarity between query and document chunks
- Returns the most relevant content based on similarity scores

### 5. Response Generation
- Selects top matching content chunks
- Filters responses based on minimum similarity threshold (0.1)
- Combines multiple relevant chunks for comprehensive answers
- Provides confidence scores for transparency

## 📊 Technical Details

### Performance Metrics
- **Processing Speed**: Handles 50,000+ characters of PDF content
- **Vocabulary Size**: Up to 5,000 unique terms
- **Response Time**: Near real-time query processing
- **Accuracy**: Similarity-based matching with confidence scoring

### Algorithm Features
- **TF-IDF Vectorization**: Captures term importance across documents
- **Cosine Similarity**: Measures semantic similarity between queries and content
- **Dynamic Filtering**: Adjusts response quality based on similarity thresholds
- **Error Handling**: Robust error management for various edge cases

## 🔍 Advanced Features

### Intelligent Text Chunking
- Processes sentences longer than 20 characters for quality content
- Maintains context while splitting documents
- Preserves semantic meaning across chunks

### Similarity Scoring
- Multi-level similarity analysis
- Top-3 response ranking system
- Confidence-based filtering
- Combined response generation from multiple relevant chunks

### User Experience
- Clean, intuitive Streamlit interface
- Real-time processing feedback
- Error messages for troubleshooting
- Responsive design for various screen sizes

## 🚨 Troubleshooting

### Common Issues

1. **"Command not found" errors**
   - Ensure virtual environment is activated: `source .venv/bin/activate`
   - Use full path: `.venv/bin/python -m streamlit run chatbot.py`

2. **Low similarity scores**
   - Check if PDF content was extracted properly
   - Ensure questions are related to the document content
   - Try rephrasing questions with keywords from the documents

3. **Import errors**
   - Verify all packages are installed: `pip list`
   - Reinstall requirements: `pip install -r requirements.txt`

4. **PDF processing issues**
   - Ensure PDF files are in the `pdf data/` directory
   - Check if PDFs contain extractable text (not just images)
   - Verify file permissions

## 📈 Future Enhancements

- [ ] Support for additional document formats (Word, txt, etc.)
- [ ] Enhanced deep learning models for better understanding
- [ ] Multi-language support
- [ ] Document summarization features
- [ ] Question-answer history tracking
- [ ] Advanced semantic search capabilities
- [ ] User feedback integration for continuous learning

## 📄 License

This project is open-source and available for educational and research purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## 📞 Support

If you encounter any issues or have questions about the implementation, please refer to the troubleshooting section above or check the error messages in the terminal output.

---

**Built with ❤️ using Python, Streamlit, and Machine Learning by Raushan**