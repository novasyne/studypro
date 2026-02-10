
<h1 align="center">STUDYPRO</h1>

<p align="center">
  <strong>Open-Source Clinical Trial Management System for Digital Biomarkers and AI-enabled Analytics</strong>
</p>

<p align="center">
  <a href="https://studypro.novasyne.com">Live Demo</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#digital-biomarkers">Digital Biomarkers</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/license-Open%20Source-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/platform-Docker%20%7C%20Cloud-lightgrey.svg" alt="Platform"/>
</p>

---

## Overview

**STUDYPRO** is a modern, lightweight clinical trial management system (CTMS) designed specifically for academic institutions, hospitals, and investigator-initiated research. Unlike expensive enterprise solutions, STUDYPRO provides a transparent, accessible platform that seamlessly integrates subject management, digital biomarker recording, and AI-powered analytics into a single unified interface.


> **Author:** Gideon Vos, James Cook University, Australia  
> **Contact:** [LinkedIn](https://www.linkedin.com/in/gideonvos)  
> **Released:** January 2026

---


**🌐 Try it now:** [https://studypro.novasyne.com](https://studypro.novasyne.com)

### Why STUDYPRO?

| Challenge | STUDYPRO Solution |
|-----------|-------------------|
| Enterprise CTMS costs $50,000–$200,000+ | **< $500/year** for infrastructure |
| Months of deployment and training | **Deploy in hours**, intuitive interface |
| No native digital biomarker support | **Built-in EEG, wearable, and imaging modules** |
| Separate tools for analytics | **Integrated AI-powered analysis** |
| Closed, proprietary systems | **Open-source**, full transparency |

---

## Features

### 📋 Study Management
- **Guided study creation wizard** with templates for Interventional, Observational, Registry, Case-Control, and Sensory Studies
- **Cohort configuration** for treatment arms, placebo groups, and control conditions
- **Customizable biomarker library** with standardized definitions, units, and collection methods
- **Role-based access control** for Principal Investigators, Coordinators, and Data Managers

### 👥 Participant & Subject Management
- Comprehensive demographics and medical history tracking
- Consent documentation and version management
- Cohort assignment and randomization tracking
- Concomitant medication logging with dosing schedules
- Activity status monitoring (active, inactive, withdrawn, completed)

### 📊 Real-Time Dashboards
- **Key Performance Indicators**: Study status, enrollment counts, recording totals, budget remaining
- **Enrollment tracking**: Cumulative accrual plots vs. target trajectories
- **Financial burn rate monitoring**: Expense projections and budget exhaustion forecasting
- **Cohort distribution summaries**: Demographics balance across study arms
- **Recording activity heatmaps**: Per-subject compliance visualization
- **Automated alerts**: Inactive subjects, unresolved adverse events, protocol deviations

### 📈 Built-In Statistical Analysis
- **Cohort comparisons**: T-tests, ANOVA, Mann-Whitney U, Kruskal-Wallis with effect sizes
- **Longitudinal tracking**: Individual subject trajectory plots with biomarker overlays
- **Correlation analysis**: Pearson/Spearman matrices with interactive heatmaps
- **Categorical analysis**: Chi-square tests, frequency distributions, cross-tabulations
- **Visualizations**: Box plots, violin plots, scatter plots, grouped bar charts

### 🤖 AI-Powered Analytics
- **Retrieval-Augmented Generation (RAG)**: Upload protocols, literature, and domain documents
- **Automated analysis reports**: AI-generated narrative interpretations of trial data
- **Citation and traceability**: All AI outputs include source references for verification
- **Knowledge base management**: Build institutional knowledge repositories

### 🔒 Regulatory Compliance
- **Comprehensive audit trails**: Immutable logs of all data modifications
- **User attribution and timestamps**: Complete provenance tracking
- **MedDRA coding**: Standardized adverse event classification
- **GCP-aligned workflows**: Designed with Good Clinical Practice principles
- **Data export**: CSV, Excel formats compatible with R, Python, SPSS, SAS

### 💰 Financial Management
- Budget establishment and amendment tracking
- Categorized expense logging with receipt attachments
- Real-time burn rate visualization
- Budget exhaustion projections

---

## Digital Biomarkers

STUDYPRO provides **native support** for digital biomarkers—a critical differentiator from legacy CTMS platforms that treat high-frequency physiological data as an afterthought.

### 🧠 Electroencephalography (EEG)


- **Supported formats**: EDF (European Data Format), BDF (BioSemi Data Format), CSV
- **Automated processing** via MNE-Python integration:
  - Channel mapping and montage recognition
  - Metadata extraction (sampling rate, channels, duration)
  - Initial quality checks
- **On-demand analysis**:
  - Band power computation (Delta, Theta, Alpha, Beta, Gamma)
  - Topographic distribution maps
  - Power spectral density (PSD) plots
  - Coherence analysis
  - Event-related potentials (ERPs)

### ⌚ Wearable Device Data

STUDYPRO ingests high-frequency time-series data from consumer and medical-grade wearables:

| Device | Supported Signals |
|--------|-------------------|
| **Empatica E4** | Heart rate, electrodermal activity (EDA), accelerometry, temperature |
| **Generic CSV** | Any time-series data from other manufacturers |
| **Custom formats** | Extensible parser architecture |

**Capabilities:**
- Heart rate variability (HRV) analysis
- Activity pattern recognition
- Sleep quality metrics
- Stress biomarker derivation

### 🏥 Medical Imaging

- **MRI** (Magnetic Resonance Imaging)
- **CT** (Computed Tomography)
- **X-ray** scans
- DICOM metadata extraction
- Secure storage with audit trails

### 🔬 Traditional Biomarkers

- **Vital signs**: Blood pressure, heart rate, temperature, respiratory rate
- **Laboratory values**: Glucose, hemoglobin, lipid panels, liver function, renal function
- **Anthropometrics**: Height, weight, BMI, body composition
- **Standardized scales**: BDI, HAM-D, GAD-7, quality-of-life instruments with automated scoring

---

## Quick Start

### Option 1: Cloud Hosted (Recommended)

The fastest way to get started is using our cloud-hosted platform:

👉 **[https://studypro.novasyne.com](https://studypro.novasyne.com)**

No installation required. Create an account and start managing your study immediately.

---

### Option 2: Local Deployment with Docker

Deploy STUDYPRO locally using Docker for full data control and offline capability.

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/studypro.git
cd studypro
```

#### Step 2: Configure Environment

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Security
SECRET_KEY=your-secure-secret-key-here

# Database
DB_SERVER=sql
DB_USER=sa
DB_PASS=YourStrongP@ssw0rd!
DB_NAME=studypro

# Email (optional)
MAIL_SERVER=smtp.your-provider.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_PASSWORD=your-mail-password

# AI Analytics (optional - requires OpenAI API key)
OPENAI_API_KEY=sk-your-openai-key
```

#### Step 3: Launch Services

```bash
docker-compose up -d
```

This starts three containers:
- **sql**: Microsoft SQL Server 2022 Express
- **azurite**: Azure Blob Storage emulator (for file storage)
- **app**: STUDYPRO Flask application

#### Step 4: Access STUDYPRO

Open your browser and navigate to:

```
http://localhost:8080
```

#### Verify Services

```bash
# Check all containers are running
docker-compose ps

# View application logs
docker-compose logs -f app

# Stop all services
docker-compose down
```

---

### Option 3: Manual Installation (Development)

For development or customization:

#### Prerequisites
- Python 3.11+
- Microsoft SQL Server (or SQL Server Express)
- ODBC Driver 17 for SQL Server

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/studypro.git
cd studypro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
export SECRET_KEY="dev-secret-key"
export DB_SERVER="localhost"
export DB_USER="sa"
export DB_PASS="YourPassword"
export DB_NAME="studypro"

# Initialize database
python init_db.py

# Run development server
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## Architecture

STUDYPRO follows a three-tier architecture optimized for clinical research workflows:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│              Flask + Jinja2 + Bootstrap                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                      │
│    Authentication │ Study Management │ Analytics Engine      │
│    Role-Based Access │ Biomarker Processing │ AI/RAG        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Persistence Layer                    │
│         SQL Server          │         Blob Storage           │
│    (Structured Data)        │    (Files, EEG, Images)        │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11, Flask 3.1 |
| **Database** | Microsoft SQL Server 2022 |
| **File Storage** | Azure Blob Storage / Azurite |
| **EEG Processing** | MNE-Python 1.10 |
| **Data Analysis** | NumPy, SciPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Bokeh |
| **AI/LLM** | LangChain, OpenAI GPT-4.5 |
| **Containerization** | Docker, Docker Compose |

---

## Cost Structure

STUDYPRO dramatically reduces the cost of clinical trial management:

### Cloud Infrastructure Costs (100-Subject Trial)

| Component | Specification | Monthly Cost |
|-----------|---------------|--------------|
| Database | SQL Database (Basic, 2GB) | $5.00 |
| File Storage | Blob Storage (100GB) | $1.80 |
| Transactions | ~100k/month | $0.10 |
| App Service | Basic (1 core, 1.75GB) | $13.14 |
| **Total** | | **$20.04/month** |

### Scaling Analysis

| Trial Size | Monthly Cost | Per-Subject | Annual Cost |
|------------|--------------|-------------|-------------|
| 50 subjects | $18 | $0.36 | $216 |
| 100 subjects | $20 | $0.20 | $240 |
| 200 subjects | $30 | $0.15 | $360 |
| 500 subjects | $75 | $0.15 | $900 |
| 1000 subjects | $150 | $0.15 | $1,800 |

### Comparison with Commercial CTMS

| Metric | STUDYPRO | Mid-Tier CTMS | Enterprise CTMS |
|--------|----------|---------------|-----------------|
| 2-Year TCO (100 subjects) | **$480–$5,480** | ~$63,000 | ~$230,000 |
| Deployment Time | Hours | 4–8 weeks | 3–6 months |
| Digital Biomarker Support | Native | Limited | Add-on |
| AI Analytics | Integrated | Basic/None | Premium Add-on |

---

## Supported Data Standards

- **MedDRA**: Medical Dictionary for Regulatory Activities for adverse event coding
- **Export formats**: CSV, Excel (compatible with R, Python, SPSS, SAS)
- **Future roadmap**: CDISC ODM, SDTM, FHIR

---

## Cloud Deployment

For production cloud deployments, STUDYPRO supports:

- **Microsoft Azure** (recommended): Azure App Service + Azure SQL + Azure Blob Storage
- **Amazon Web Services**: Elastic Beanstalk + RDS + S3
- **Google Cloud Platform**: Cloud Run + Cloud SQL + Cloud Storage

---

## Use Cases

STUDYPRO is ideal for:

- **Academic clinical trials** with limited budgets
- **Investigator-initiated studies** requiring rapid deployment
- **Digital biomarker research** involving EEG, wearables, or continuous monitoring
- **Neurophysiology studies** needing integrated signal analysis
- **Multi-site collaborations** requiring centralized data management
- **Pilot studies** validating novel endpoints before larger trials

---

## License

STUDYPRO is released as open-source software. Organizations can deploy, modify, and maintain the platform independently. Citation is required for academic and commercial use.

---

## Support & Contact

- **Cloud Platform**: [https://studypro.novasyne.com](https://studypro.novasyne.com)
- **Company Website**: [https://www.novasyne.com](https://www.novasyne.com)
- **Author**: Gideon Vos, James Cook University
- **LinkedIn**: [https://www.linkedin.com/in/gideonvos](https://www.linkedin.com/in/gideonvos)
- **Issues**: [GitHub Issues](https://github.com/novasyne/studypro/issues)

---

## Acknowledgments

STUDYPRO was developed with the goal of democratizing access to modern clinical trial management for resource-limited research settings worldwide.

---

<p align="center">
  <strong>Built for researchers, by researchers.</strong>
</p>

<p align="center">
  <a href="https://studypro.novasyne.com">Get Started →</a>
</p>
