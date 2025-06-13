# SEO Analysis Dashboard

An AI-powered dashboard for analyzing Google Ads keyword performance and generating SEO suggestions.

## Features

- 📊 Interactive keyword performance analysis
- 🔍 Automatic keyword clustering
- 🤖 AI-powered keyword suggestions
- 📝 Meta description generation
- 📤 Export functionality

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally (for LLM features)
- Google Ads API access (optional, for live data)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/seo-insight.git
cd seo-insight
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
```

5. Start Ollama (if using local LLM):

```bash
ollama serve
```

## Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Navigate to the dashboard in your browser (default: http://localhost:8501)

3. Choose your data source:

   - Upload a CSV file with keyword data
   - Use mock data for testing

4. Analyze keywords:
   - View performance metrics
   - Adjust clustering parameters
   - Generate AI suggestions
   - Export results

## Data Format

The dashboard expects CSV files with the following columns:

- keyword
- impressions
- clicks
- cost
- conversions
- avg_position

## Development

### Project Structure

```
seo-insight/
├── app.py              # Main Streamlit application
├── pages/             # Streamlit pages
│   └── dashboard.py   # Main dashboard
├── services/          # External service integrations
│   ├── google_ads.py  # Google Ads API
│   └── llm_generator.py # LLM integration
├── utils/             # Utility functions
│   ├── auth.py        # Authentication
│   └── keyword_utils.py # Keyword analysis
└── data/             # Data storage
    └── mock_keyword_data.csv
```

### Adding Features

1. Create new Streamlit pages in the `pages/` directory
2. Add service integrations in `services/`
3. Implement utility functions in `utils/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## TODO

- [ ] Add unit tests
- [ ] Implement caching for better performance
- [ ] Add more visualization options
- [ ] Support for additional data sources
