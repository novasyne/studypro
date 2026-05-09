# (c) Novasyne 2025. www.novasyne.com or studypro.novasyne.com

import itertools
import json
import math
import os
import random
import re
import secrets
import threading
import uuid
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from io import BytesIO
from urllib.parse import quote_plus
import numpy as np
_rng = np.random.default_rng()
import openai
import pandas as pd
import pydicom
import nibabel as nib
from PIL import Image
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, jsonify, send_file, g
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from flask_mail import Message
from flask_wtf.csrf import CSRFProtect
from itsdangerous import SignatureExpired, BadTimeSignature, BadSignature
from markupsafe import escape as html_escape
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import (
    BlobServiceClient, BlobSasPermissions, generate_blob_sas
)
from sqlalchemy import func, case, or_, desc
from sqlalchemy.orm import aliased
from scipy.stats import (
    ttest_ind, f_oneway, mannwhitneyu, kruskal,
    pearsonr, spearmanr, gaussian_kde
)
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.preprocessing import StandardScaler
from langchain_openai import OpenAIEmbeddings
from bokeh.embed import components, json_item
from bokeh.models import (
    ColumnDataSource, HoverTool, NumeralTickFormatter, FactorRange,
    LinearColorMapper, ColorBar, LabelSet, Span
)
from bokeh.palettes import (
    Viridis256, Category10, Category20, Category20c
)
from bokeh.plotting import figure
from bokeh.transform import cumsum
import extensions
from extensions import db, mail, login_manager, initialize_serializer
from models import (
    User, Study, StudyParticipantLink, StudyType,
    StudyParticipant, StudyArm, Subject, SubjectClinician,
    SubjectConsent, SubjectContact, SubjectDiagnosis, SubjectMedication,
    StudyDocument, SubjectDocument, StudyRecording, BiomarkerType,
    StudyRecordingBiomarker, StudyRecordingEEG, StudyRecordingWearable,
    EEG, Wearable, FinancialLedger, ExpenseCategory, AuditLog,
    StudySettings, StudyKnowledge, StudyKnowledgeVector,
    MedDRA, SubjectSymptom, SubjectAdverseEvent, SubjectMedicationTaken,
    StudyRecordingImage, study_settings_biomarker_types
)

import mne
import tempfile
import warnings
from urllib.parse import urlparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import requests

from flask import request, redirect, url_for, flash, render_template
from flask_mail import Message

warnings.filterwarnings("ignore", category=UserWarning)

class ClinicalTrialsService:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_studies(self, condition=None, intervention=None, status=None, limit=100):
        params = {
            "format": "json",
            "pageSize": limit,
            "fields": "NCTId,Condition,BriefTitle,EnrollmentCount,StartDate,CompletionDate,OverallStatus,Phase"
        }

        query_parts = []

        if condition:
            query_parts.append(self._build_area_clause("Condition", condition))

        if intervention:
            query_parts.append(self._build_area_clause("Intervention", intervention))

        if status:
            status = status.upper().replace(" ", "_")
            query_parts.append(f"AREA[OverallStatus]({status})")

        params["query.term"] = " AND ".join([q for q in query_parts if q])

        response = requests.get(self.BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        return self._process_response(response.json())

    def _build_area_clause(self, field_name, value):
        if not value:
            return None
        if isinstance(value, list):
            terms = [f'"{v}"' for v in value if v]
            return f"AREA[{field_name}]({' OR '.join(terms)})"
        return f'AREA[{field_name}]("{value}")'

    def _process_response(self, data):
        studies = data.get('studies', [])
        processed = []

        for study in studies:
            protocol = study.get('protocolSection', {})
            ident = protocol.get('identificationModule', {})
            status_mod = protocol.get('statusModule', {})
            design = protocol.get('designModule', {})

            processed.append({
                'nct_id': ident.get('nctId'),
                'title': ident.get('briefTitle'),
                'status': status_mod.get('overallStatus'),
                'start_date': self._parse_date(
                    status_mod.get('startDateStruct', {}).get('date')
                ),
                'completion_date': self._parse_date(
                    status_mod.get('completionDateStruct', {}).get('date')
                ),
                'phase': (design.get('phases') or ['N/A'])[0],
                'enrollment': design.get('enrollmentInfo', {}).get('count')
            })

        df = pd.DataFrame(processed)

        if not df.empty:
            df['duration_months'] = (
                df['completion_date'] - df['start_date']
            ) / pd.Timedelta(days=30.44)

        return df

    def _parse_date(self, date_str):
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y-%m")
            except ValueError:
                return None


HELP_CONTENT = {
    'create_study_step1': {
        'title': 'Step 1: Select Study Type & Settings',
        'content': """
        <p>This is the first step to creating a new clinical study in the system.</p>
        
        <h5>Study Type</h5>
        <p>The study type determines the structure and template for your study, including predefined study arms and other settings. It helps in standardizing studies of a similar nature.</p>
        <ul>
            <li><strong>Interventional:</strong> Studies where researchers assign participants to receive a specific intervention, such as a new drug or therapy.</li>
            <li><strong>Observational:</strong> Studies where researchers observe outcomes without manipulating interventions.</li>
            <li><strong>Registry:</strong> Studies that collect data on patients with a specific condition or exposure over time.</li>
        </ul>
        <p>Please select the category and specific type that best fits your research protocol.</p>
        
        <hr>

        <h5>Study Settings</h5>
        <p>These settings configure specific features for your study. They can be changed later from the 'Manage Study' page.</p>
        
        <p><strong>Enable AI Features:</strong></p>
        <ul>
            <li>Checking this box enables AI-powered features, such as automated analysis summaries.</li>
            <li>If checked, you must provide a valid <strong>OpenAI API Key</strong> for the features to function. This key is stored securely per-study.</li>
        </ul>

        <p><strong>Allowed Data Types:</strong></p>
        <ul>
            <li>These checkboxes act as filters for the biomarker list below.</li>
            <li><strong>Allow EEG Data:</strong> Enables 'EEG' type biomarkers in the list.</li>
            <li><strong>Allow Wearable Data:</strong> Enables 'Wearable' type biomarkers in the list.</li>
            <li><strong>Allow Biological Samples:</strong> Enables types like 'Blood', 'Saliva', 'Urine'.</li>
            <li><strong>Allow Scales:</strong> Enables questionnaire or scale-based biomarkers (e.g., 'PHQ-9', 'GAD-7').</li>
        </ul>

        <p><strong>Select Allowed Biomarkers:</strong></p>
        <ul>
            <li>This list shows all biomarkers available in the system, filtered by the 'Allowed Data Types' you just selected.</li>
            <li>Select all biomarkers you intend to record for this study (use Ctrl/Cmd + Click to select multiple).</li>
            <li>This ensures that users performing data entry can only select from the relevant list of biomarkers for this specific study.</li>
        </ul>

        <hr>

        <p>Click 'Next' to proceed to the next step. Click 'Cancel' to return to the dashboard.</p>
        """
    },
    'create_study_step2': {
        'title': 'Step 2: Define Study Arms',
        'content': """
        <p>In this step, you will define the different participant groups for your study. These are often referred to as 'arms'.</p>
        <p>Based on the Study Type you selected in the previous step, a set of default arms may have been pre-populated for you. You can customize these as needed.</p>
        <p><strong>Group Name:</strong> A short, descriptive name for the arm (e.g., "Control Group", "Treatment Group A"). This is a required field.</p>
        <p><strong>Description:</strong> A brief explanation of the arm's purpose or the intervention they will receive (e.g., "Receives placebo", "Receives 50mg of the trial drug").</p>
        <p><strong>Actions:</strong></p>
        <ul>
            <li><strong>Add Group:</strong> Click this to add a new, empty row to the table to define another arm.</li>
            <li><strong>Remove:</strong> Click this to delete a specific arm row.</li>
        </ul>
        <p>You must have at least one study arm defined to proceed. Once you are satisfied with the arms, click 'Next'.</p>
        """
    },
    'create_study_step3': {
        'title': 'Step 3: Details & Participants',
        'content': """
        <p>This is the final step for creating your study. Here you will provide the main details and add the research personnel involved.</p>
        
        <h5>Study Details</h5>
        <p><strong>Study Name:</strong> The official title of the study. This should be clear and descriptive.</p>
        <p><strong>Study Description:</strong> A more detailed summary of the study's purpose, objectives, and methods.</p>

        <hr>

        <h5>Participants</h5>
        <p>Add all the key personnel involved in conducting the study.</p>
        <ul>
            <li>Click the <strong>'Add Participant'</strong> button to open a form to enter participant details.</li>
            <li>You must add at least one participant.</li>
            <li>One participant must be assigned the role of <strong>'Principal Investigator'</strong>. This is the lead researcher for the study.</li>
            <li>You can <strong>Edit</strong> or <strong>Remove</strong> participants from the table.</li>
        </ul>
        <p>After filling in all the details and adding the participants, click <strong>'Save Study'</strong> to create the study and be redirected to the study's dashboard.</p>
        """
    },
    'edit_arms': {
        'title': 'Help: Editing Arms',
        'content': """
            <h4>Managing Study Arms</h4>
            <p>This page allows you to manage the participant groups (arms) for your study. An arm is a specific group of subjects who are treated similarly in a trial, such as a 'Control Group' or 'Treatment Group'.</p>
            <hr>
            <h6>How to Use This Page:</h6>
            <ul>
                <li><strong>View Arms:</strong> The table displays all current arms for this study.</li>
                <li><strong>Edit an Arm:</strong> You can directly edit the <strong>Arm Name</strong> and <strong>Description</strong> in the text fields within the table.</li>
                <li><strong>Add a New Arm:</strong> Click the <strong>Add Arm</strong> button to add a new row to the table. Fill in the details for your new arm.</li>
                <li><strong>Remove an Arm:</strong> Click the <strong>Remove</strong> button next to an arm to delete it. Please be careful, as this action cannot be undone.</li>
            </ul>
            <p><strong>Important:</strong> After making any changes (adding, editing, or removing), you must click the <strong>Save Changes</strong> button to apply them to the study. Clicking <strong>Cancel</strong> will discard all your changes.</p>
        """
    },
    'manage_participants': {
        'title': 'Help: Managing Participants',
        'content': """
            <h4>Managing Study Participants (Staff)</h4>
            <p>This page is for managing the research team and staff associated with your study. These are the individuals conducting the research, not the subjects being studied.</p>
            <hr>
            <h6>Key Functions:</h6>
            <ul>
                <li><strong>Add Participant:</strong> Click this button to open a detailed form where you can add a new staff member. You can enter their name, contact details, role in the study, affiliation, and financial information.</li>
                <li><strong>Edit a Participant:</strong> In the 'Actions' column of the table, click the <strong>Edit</strong> button. This will open the same form, pre-filled with the participant's current information, allowing you to make updates.</li>
                <li><strong>Remove a Participant:</strong> Click the <strong>Remove</strong> button in the 'Actions' column to disassociate a staff member from this study. This does not delete them from the system, but simply removes their link to this specific trial.</li>
            </ul>
            <p>The table provides a clear overview of all staff involved, their roles, and contact information. You can sort the table by clicking on the column headers.</p>
            <p>Click <strong>Save</strong> to confirm any removals, or click <strong>Cancel</strong> to return to the previous page.</p>
        """
    },
    'manage_subjects': {
        'title': 'Help: Managing Subjects',
        'content': """
            <h4>Managing Study Subjects</h4>
            <p>This is a central hub for managing the subjects (e.g., patients, volunteers) enrolled in your clinical trial. All data collection and management for an individual starts here.</p>
            <hr>
            <h6>Key Functions:</h6>
            <ul>
                <li><strong>Add Subject:</strong> Click this button to open the subject creation form. The <strong>External Subject Code</strong> is a crucial, unique identifier used for de-identification. You can also assign the subject to an arm and fill in their demographic and baseline health information.</li>
                <li><strong>Actions Column:</strong> Each subject has a set of management tools:
                    <ul>
                        <li><strong>Edit:</strong> Modify the subject's core demographic or baseline data.</li>
                        <li><strong>History:</strong> A comprehensive section to log and view a subject's medical history, including associated clinicians, diagnoses, medications, and emergency contacts.</li>
                        <li><strong>Consent:</strong> Manage and track the subject's consent forms. You can add new consent versions, record when they were signed, and when consent was withdrawn.</li>
                        <li><strong>Documents:</strong> Upload, view, and manage any documents relevant to the subject, such as scanned forms or lab results.</li>
                    </ul>
                </li>
            </ul>
            <p>This page is designed to give you a complete and detailed overview of every individual participating in your study.</p>
        """
    },
    'analytics': {
        'title': 'Help: Study Analytics',
        'content': """
            <h4>Running Data Analysis</h4>
            <p>This page provides powerful tools to analyze data collected during your study. The process involves selecting an analysis type, choosing the relevant data, and running the analysis to generate plots or statistical summaries.</p>
            <hr>
            <h6>Analysis Workflow:</h6>
            <ol>
                <li><strong>Select Analysis Type:</strong> Choose the primary goal of your analysis. The form will change based on your selection.
                    <ul>
                        <li><strong>(Biomarker) Arm Comparison:</strong> Compares a single biomarker across two or more arms (e.g., Treatment vs. Control).</li>
                        <li><strong>(Biomarker) Subject Analysis Over Time:</strong> Tracks the values of a single biomarker for one or more subjects over the duration of the study.</li>
                        <li><strong>(Biomarker) Data Distribution:</strong> Visualizes the spread and distribution of a single biomarker's data (e.g., with a histogram).</li>
                        <li><strong>(Biomarker) Correlation Analysis:</strong> Examines the relationship between two or more different biomarkers (a pairwise statistical report).</li>
                        <li><strong>Demographics Summary:</strong> Generates a bar chart showing the frequency of subjects by a specific demographic field (e.g., Gender, Race, Ethnicity).</li>
                        <li><strong>Event/Symptom Frequency:</strong> Creates a bar chart of the most frequently reported Adverse Events or Symptoms (using MedDRA terms).</li>
                        <li><strong>Concomitant Medication Frequency:</strong> Generates a bar chart of the most frequently logged concomitant medications.</li>
                        <li><strong>Arm Distribution Report (New!):</strong> Generates a baseline characteristics report comparing the distribution of demographics, diagnosis history, and medication history across all study arms.</li>
                        <li><strong>Comprehensive Correlation (New!):</strong> Performs a broad correlation analysis between all data types: demographics, mean biomarkers, and counts of symptoms, AEs, and medications.</li>
                        <li><strong>Automated Analysis (AI):</strong> Uses an AI model to provide a narrative summary and interpretation of all study data, optionally filtered by arm.</li>
                        <li><strong>AI Knowledge Base (New!):</strong> When 'Automated Analysis (AI)' is selected, a <strong>'Manage AI Knowledge'</strong> button will appear. This allows you to upload PDF documents (e.g., study protocols, relevant publications). After uploading, you must click <strong>'Build Store'</strong> in the modal. This creates a secure vector store of your documents. When you 'Run Analysis', the AI will now use this knowledge base to provide a much richer, context-aware interpretation.</li>
                    </ul>
                </li>
                <li><strong>Select Data:</strong>
                    <ul>
                        <li>For <strong>Biomarker</strong> analyses, the 'Biomarker(s)' box will appear. Select the data you want to analyze. The help text will tell you how many are required.</li>
                        <li>For <strong>Categorical</strong> analyses (Demographics, Events, Meds, Reports), the 'Biomarker(s)' box will be hidden.</li>
                    </ul>
                </li>
                <li><strong>Configure Options:</strong> Additional options will appear based on your selections, allowing you to choose specific arms, subjects, demographic fields, or statistical tests.</li>
                <li><strong>Run Analysis:</strong> The 'Run Analysis' button will become active once you have provided all the necessary information. Click it to generate the result in a pop-up window.</li>
            </ol>
        """
    },
    'dashboard': {
        'title': 'Help: Main Dashboard',
        'content': """
            <h4>Your Studies Dashboard</h4>
            <p>This is your main dashboard. It provides a complete overview of all the clinical trials you are associated with.</p>
            <hr>
            <h6>Key Features:</h6>
            <ul>
                <li><strong>Studies Table:</strong> Lists all your studies with key information like status, start date, and the number of enrolled subjects. You can click on the column headers to sort the data.</li>
                <li><strong>Register New Study:</strong> Click this button to launch the step-by-step wizard for creating a new study.</li>
            </ul>
            <h6>Actions Column:</h6>
            <p>Each study in the table has a set of quick-action buttons:</p>
            <ul>
                <li><strong>Manage:</strong> Takes you to the central management page for the study, where you can edit details, manage staff, and access all other study components.</li>
                <li><strong>Analytics:</strong> Opens the data analytics suite for that study.</li>
                <li><strong>Recordings:</strong> Jumps directly to the data recordings page to log or view subject data.</li>
                <li><strong>Documents:</strong> Opens a modal to manage study-level documents, such as the protocol, ethics approval, or investigator brochures.</li>
            </ul>
        """
    },
    'edit_study': {
        'title': 'Help: Manage Study',
        'content': """
            <h4>Study Management Hub</h4>
            <p>This page is the central hub for managing the core details and components of your study.</p>
            <hr>
            <h6>Study Details:</h6>
            <p>You can update the primary information for your study here, including its name, description, dates, status, and funding details. Click the <strong>Update</strong> button to save any changes you make.</p>
            <h6>Navigation:</h6>
            <p>Use the buttons at the bottom to navigate to the different management areas of your study:</p>
            <ul>
                <li><strong>Participants:</strong> Manage the research staff and team members.</li>
                <li><strong>Arms:</strong> Define and edit the study arms (e.g., Treatment, Control).</li>
                <li><strong>Subjects:</strong> Add and manage the individuals enrolled in the trial.</li>
                <li><strong>Finances:</strong> Track the study's budget and expenses.</li>
                <li><strong>Settings:</strong> Opens a modal to configure study-specific settings, such as enabling AI features or managing the list of allowed data types (Biomarkers, Scales, EEG, etc.) for data entry.</li>
            </ul>
            <h6>Advanced Actions:</h6>
            <ul>
                <li><strong>Export Study:</strong> This will compile all study data—including details, subjects, recordings, and documents—into a single downloadable ZIP file.</li>
                <li><strong>Revoke Study:</strong> <strong class="text-danger">Use with extreme caution.</strong> This action will permanently delete the study and all of its associated data. This cannot be undone.</li>
            </ul>
        """
    },
    'finances': {
        'title': 'Help: Financial Ledger',
        'content': """
            <h4>Tracking Study Finances</h4>
            <p>This page provides a comprehensive overview of the study's financial status, allowing you to track the budget, log expenses, and monitor spending.</p>
            <hr>
            <h6>Summary Cards:</h6>
            <p>The cards at the top give you an at-a-glance summary:</p>
            <ul>
                <li><strong>Total Budget:</strong> The initial budget plus any additional funds (top-ups).</li>
                <li><strong>Total Expenses:</strong> The sum of all logged expenses.</li>
                <li><strong>Remaining Balance:</strong> The difference between the total budget and expenses.</li>
            </ul>
            <h6>Managing Transactions:</h6>
            <ul>
                <li><strong>Add Expense:</strong> Click to open a form where you can record a new expense. You must provide a date, category, description, and amount.</li>
                <li><strong>Top-up Budget:</strong> Click to add new funds to the study's budget. This is useful for recording additional grants or funding tranches.</li>
            </ul>
            <h6>Transaction History:</h6>
            <p>The main table lists every financial transaction in chronological order. The progress bar at the bottom provides a visual representation of how much of the total budget has been spent.</p>
        """
    },
    'study_recordings': {
        'title': 'Help: Study Recordings',
        'content': """
            <h4>Managing Data Recordings</h4>
            <p>This page is where you log and manage all data points collected from subjects throughout the study. This includes biomarker values, symptoms, adverse events, medications, and file-based data.</p>
            <hr>
            <h6>Key Functions:</h6>
            <ul>
                <li><strong>New Recording:</strong> Click this to open the all-in-one data entry form.
                    <ol>
                        <li>Select the subject and the date/time of the recording.</li>
                        <li>Choose the <strong>Recording Type</strong>. The form will dynamically change to accept the correct data.</li>
                        <li>For <strong>Biomarker</strong>, select the biomarker type (e.g., from 'Blood', 'Saliva') and enter the numerical value.</li>
                        <li>For <strong>Scale</strong>, select the scale type (e.g., 'PHQ-9') and enter the numerical score.</li>
                        <li>For <strong>Symptom</strong> or <strong>Adverse Event</strong>, you can enter the patient's verbatim report and then use the <strong>MedDRA search</strong> to select a standardized term. You must also record severity or grade.</li>
                        <li>For <strong>Medication</strong>, log the medication name, dose, route, and indication (reason for taking).</li>
                        <li>For <strong>EEG</strong> or <strong>Wearable</strong>, select the device used and upload the corresponding data file(s). Multiple files will be automatically zipped.</li>
                    </ol>
                </li>
                <li><strong>Calendar View:</strong> Provides a monthly calendar to visualize the frequency of data collection. Clicking on a date will show all recordings from that day.</li>
                <li><strong>Import/Export:</strong>
                    <ul>
                        <li>Click <strong>Download Template</strong> to get an Excel file pre-filled with the allowed biomarker/scale columns for this study.</li>
                        <li>Fill in the template with your data and click <strong>Import Template</strong> to bulk-upload biomarker records.</li>
                        <li>The table also includes buttons to <strong>Export</strong> the current view to CSV or Excel.</li>
                    </ul>
                </li>
                <li><strong>Actions:</strong> For each recording, you can:
                    <ul>
                        <li><strong>Visualize:</strong> For EEG recordings, click this to open the EEG visualization and analysis tool.</li>
                        <li><strong>Download:</strong> Download the associated raw file (if applicable).</li>
                        <li><strong>Delete:</strong> Permanently delete the entry.</li>
                    </ul>
                </li>
            </ul>
        """
    },
    'audit_log': {
        'title': 'Help: Audit Log',
        'content': """
            <h4>Audit Log Viewer</h4>
            <p>This page displays a chronological record of all changes made to sensitive data, specifically all recordings (Biomarker, EEG, Wearable, Symptom, Adverse Event, and Medication).</p>
            <hr>
            <h6>Key Features:</h6>
            <ul>
                <li><strong>Log Table:</strong> Shows who made the change, what type of data was affected (e.g., Biomarker), the operation (Create, Update, Delete), and the associated Study/Subject.</li>
                <li><strong>View Details:</strong> Click the 'View' button to open a popup showing the exact data that was changed. It displays the 'Old Value' and 'New Value' in a side-by-side comparison.</li>
                <li><strong>Clear Log:</strong> <strong class="text-danger">This is a permanent action.</strong> It will delete all entries from the audit log. This cannot be undone.</li>
                <li><strong>Close:</strong> Returns you to the main dashboard.</li>
            </ul>
        """
    },
    'manage_account': {
        'title': 'Help: Manage Account',
        'content': """
            <h4>Account Management</h4>
            <p>This page allows you to manage your personal user details.</p>
            <hr>
            
            <h5>Your Details</h5>
            <p>You can update your personal information here, including your name, email, and contact details. Click <strong>'Update Details'</strong> to save any changes.</p>
            <ul>
                <li><strong>Email Change:</strong> If you change your email address, it will be updated for both your login and your participant profile. Ensure it is a valid, unique email.</li>
            </ul>

            <h5>Change Your Password</h5>
            <p>This form is only visible when you are viewing your own profile. You can change your own password by providing your old password and a new password that meets the strength requirements (at least 8 characters, 1 uppercase, 1 lowercase, and 1 number).</p>

            <hr>

            <h5>Administrator Functions</h5>
            <p>If your role is an Administrator, you will see additional options:</p>
            <ul>
                <li><strong>User Selection:</strong> A dropdown menu appears at the top, allowing you to select and manage any user in the system.</li>
                <li><strong>Full Edit Access:</strong> You can edit all fields for any user, including their assigned 'Role'.</li>
                <li><strong>Role Change Rule:</strong> A Principal Investigator cannot change their own role unless at least one other Principal Investigator is assigned in the system.</li>
                <li><strong>Audit Log:</strong> A link to the 'Audit Log' page is available at the bottom, allowing you to review all system-wide data changes.</li>
            </ul>
        """
    },
    'study_dashboard': {
        'title': 'Help: Study Status',
        'content': """
            <p>This is the <strong>Study Status Dashboard</strong>, a high-level visual overview of your study's progress and health. It summarizes key metrics from enrollment, data collection, and finance.</p>
            <hr>
            
            <h4>Key Performance Indicators (KPIs)</h4>
            <p>The cards at the top provide an at-a-glance summary of the most important metrics:</p>
            <ul>
                <li><strong>Status:</strong> The current operational status of the study (e.g., Planned, Active, Completed).</li>
                <li><strong>Active / Total Subjects:</strong> The number of subjects currently enrolled (not withdrawn) versus the total number ever enrolled.</li>
                <li><strong>Total Recordings Logged:</strong> The total count of all data points (Biomarker, EEG, Wearable, Symptom, AE, etc.) collected.</li>
                <li><strong>Remaining Budget:</strong> The current financial balance, calculated from the Financial Ledger.</li>
            </ul>
            <hr>

            <h4>Visual Dashboards</h4>
            <p>The dashboard features several charts to help you visualize trends (charts will only appear if data is available):</p>
            <ul>
                <li><strong>Financial Burn Rate vs Budget:</strong> Tracks your cumulative expenses over time against your total budget.</li>
                <li><strong>Cumulative Subject Enrollment:</strong> Shows the rate at which subjects have been consented into the study over time.</li>
                <li><strong>Subject Distribution by Arm:</strong> A donut chart showing how your enrolled subjects are distributed across the different arms (e.g., Treatment vs. Control).</li>
                <li><strong>Biomarker Recordings Total:</strong> A bar chart displaying the total number of recordings collected for each specific biomarker.</li>
                <li><strong>Recording Activity Heatmap:</strong> A grid showing data collection activity (number of recordings) per subject, per week. This is useful for identifying gaps in data collection or monitoring subject compliance.</li>
            </ul>
            <hr>

            <h4>Subject Events & Flags</h4>
            <p>This card automatically identifies subjects that may require attention and summarizes recent critical events:</p>
            <ul>
                <li><strong>Flagged Subjects:</strong> Identifies subjects who are <span class="badge bg-danger">Withdrawn</span> or <span class="badge bg-warning text-dark">Inactive</span> (no new data in >30 days).</li>
                <li><strong>Recent Adverse Events:</strong> Shows a list of the most recently reported Adverse Events, prioritized by severity.</li>
                <li><strong>Recent Symptoms:</strong> Displays the most recently reported symptoms from subjects.</li>
            </ul>
        """
    },
    'default': {
        'title': 'Help',
        'content': '<p>No specific help content is available for this page. Please refer to the system documentation or contact support.</p>'
    }
}

FAKE_FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
]

FAKE_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

DEMO_STUDY_DATA = {
    'rct': {
        'name': 'Effect of Omega-3 Supplementation on Cardiovascular Inflammation',
        'description': 'Participants are randomly assigned to receive omega-3 or placebo to measure inflammation reduction.',
        'duration_months': 9,
        'arms': [
            {'arm_name': 'Treatment Group', 'description': 'Participants receiving 500mg Omega-3 supplementation daily.'},
            {'arm_name': 'Control Group', 'description': 'Participants receiving a daily placebo.'}
        ],
        'biomarkers': [
            {'name': 'CRP', 'min': 0.1, 'max': 10.0},
            {'name': 'TNF-a', 'min': 1.0, 'max': 25.0},
            {'name': 'Troponin I/T', 'min': 0.0, 'max': 0.4}
        ],
        'symptoms': [],
        'adverse_events': [],
        'medications': [
            {'name': 'Ibuprofen', 'dose': '200 mg', 'indication': 'Headache'},
            {'name': 'Aspirin', 'dose': '81 mg', 'indication': 'Cardioprotection'}
        ],
        'financials': {
            'topups': [
                {'description': 'Q2 Funding Tranche', 'amount': 20000}
            ],
            'expenses': [
                {'category': 'Participant Reimbursement', 'description': 'Initial travel stipends', 'amount': 2500},
                {'category': 'Lab Supplies', 'description': 'Biomarker assay kits', 'amount': 8000},
                {'category': 'Pharmacy Services', 'description': 'Placebo/Omega-3 preparation', 'amount': 3500},
                {'category': 'Data Management', 'description': 'Database setup fee', 'amount': 1500},
                {'category': 'Ethics Committee Review', 'description': 'Annual review fee', 'amount': 500}
            ]
        }
    },
    'cohort': { 
        'name': 'Longitudinal Analysis of Ferritin Levels and Anemia Risk in Adults',
        'description': 'Follows adults over time to assess relationship between ferritin and anemia development.',
        'duration_months': 84,
        'arms': [
            {'arm_name': 'Male Arm', 'description': 'Male participants enrolled in the longitudinal study.'},
            {'arm_name': 'Female Arm', 'description': 'Female participants enrolled in the longitudinal study.'}
        ],
        'biomarkers': [
            {'name': 'Ferritin', 'min': 30.0, 'max': 400.0},
            {'name': 'Hemoglobin', 'min': 12.0, 'max': 17.0},
            {'name': 'CRP', 'min': 0.1, 'max': 10.0}
        ],
        'symptoms': [],
        'adverse_events': [],
        'medications': [
            {'name': 'Ferrous Sulfate', 'dose': '325 mg', 'indication': 'Iron Deficiency'}
        ],
        'financials': {
            'topups': [
                {'description': 'Year 3 Funding Renewal', 'amount': 15000},
                {'description': 'Year 5 Funding Renewal', 'amount': 15000}
            ],
            'expenses': [
                {'category': 'Participant Reimbursement', 'description': 'Annual participant stipends (Year 1)', 'amount': 4000},
                {'category': 'Lab Supplies', 'description': 'Annual blood draw kits (Year 1)', 'amount': 2000},
                {'category': 'Data Management', 'description': 'Long-term data storage costs (Year 1)', 'amount': 1000},
                {'category': 'Participant Reimbursement', 'description': 'Annual participant stipends (Year 2)', 'amount': 4000},
                {'category': 'Lab Supplies', 'description': 'Annual blood draw kits (Year 2)', 'amount': 2200}
            ]
        }
    },
    'case-control': {
        'name': 'Association of TNF-α with Rheumatoid Arthritis Incidence',
        'description': 'Compares RA patients vs. healthy controls to assess inflammatory biomarker differences.',
        'duration_months': 12,
        'arms': [
            {'arm_name': 'Case Group (RA)', 'description': 'Patients diagnosed with Rheumatoid Arthritis.'},
            {'arm_name': 'Control Group (Healthy)', 'description': 'Healthy individuals matched for age and gender.'}
        ],
        'biomarkers': [
            {'name': 'TNF-a', 'min': 1.0, 'max': 25.0},
            {'name': 'IL-6', 'min': 0.5, 'max': 15.0},
            {'name': 'CRP', 'min': 0.1, 'max': 10.0}
        ],
        'symptoms': [],
        'adverse_events': [],
        'medications': [
            {'name': 'Methotrexate', 'dose': '15 mg/week', 'indication': 'RA'},
            {'name': 'Prednisone', 'dose': '5 mg', 'indication': 'Inflammation'}
        ],
        'financials': {
            'topups': [],
            'expenses': [
                {'category': 'Subject Recruitment', 'description': 'Advertisements and screening costs for RA patients', 'amount': 6000},
                {'category': 'Lab Supplies', 'description': 'RA panel assay kits (TNF-a, IL-6)', 'amount': 12000},
                {'category': 'Participant Reimbursement', 'description': 'Stipends for case/control groups', 'amount': 3000},
                {'category': 'Ethics Committee Review', 'description': 'Initial submission fee', 'amount': 1000}
            ]
        }
    },
    'food_sensory': {
        'name': 'Sensory Analysis of a Novel Umami Flavor Additive (F-22b)',
        'description': 'A double-blind, placebo-controlled trial to evaluate the taste, aroma, and safety profile of a new food additive.',
        'duration_months': 1,
        'arms': [
            {'arm_name': 'Additive F-22b (5mg)', 'description': 'Participants consuming a standardized meal with 5mg of Additive F-22b.'},
            {'arm_name': 'Placebo Control', 'description': 'Participants consuming a standardized meal with a placebo additive.'}
        ],
        'biomarkers': [
            {'name': 'Taste Score (1-10)', 'min': 1.0, 'max': 10.0},
            {'name': 'Aroma Rating (1-10)', 'min': 1.0, 'max': 10.0},
            {'name': 'Aftertaste Severity (1-10)', 'min': 0.0, 'max': 10.0}
        ],
        'symptoms': [],
        'adverse_events': [],
        'medications': [
            {'name': 'Antacid (Tums)', 'dose': '2 tablets', 'indication': 'Indigestion'}
        ],
        'financials': {
            'topups': [],
            'expenses': [
                {'category': 'Participant Reimbursement', 'description': 'Participant stipends for taste test sessions', 'amount': 5000},
                {'category': 'Lab Supplies', 'description': 'Food additive (F-22b) and placebo samples', 'amount': 3000},
                {'category': 'Data Management', 'description': 'Sensory data analysis software license', 'amount': 1000},
                {'category': 'Site Costs', 'description': 'Rental of sensory lab space', 'amount': 1500}
            ]
        }
    }
}

SYNTHESIS_CORR_THRESHOLD = 0.8
SYNTHESIS_NOISE_LEVEL = 0.05
EPOCH_DURATION_S = 10

CONDITION_MAP = {
    "Stressed": ("Aggregated_Stress.csv", "Stress", 1),
    "Relaxed": ("Aggregated_Stress.csv", "Stress", 0),
    "Normal": ("Aggregated_Stress.csv", "Stress", 0),
    "Low MW": ("Aggregated_MW.csv", "Low", 1),
    "Medium MW": ("Aggregated_MW.csv", "Medium", 1),
    "High MW": ("Aggregated_MW.csv", "High", 1),
    "Relax Task": ("Aggregated_SAM.csv", "Relax", 1),
    "Arithmetic": ("Aggregated_SAM.csv", "Arithmetic", 1),
    "Stroop": ("Aggregated_SAM.csv", "Stroop", 1),
    "Mirror Image": ("Aggregated_SAM.csv", "Mirror image", 1),
}

TRIAL_TYPE_TO_EEG_MAP = {
    'rct': {
        'default': 'Relaxed' # Standard trial, assume relaxed baseline
    },
    'arm': {
        'default': 'Relaxed' # Observational, assume relaxed baseline
    },
    'case-control': {
        'Case Group (RA)': 'Stressed', # Use "Stressed" as proxy for pain/inflammation
        'Control Group (Healthy)': 'Relaxed',
        'default': 'Relaxed'
    },
    'food_sensory': {
        'default': 'Arithmetic' # Use "Arithmetic" as proxy for a high-focus task
    }
}

EEG_BIOMARKER_NAMES = [
    "Peak Alpha Frequency",
    "Relative Band Power – Alpha (8–13 Hz)",
    "Relative Band Power – Theta (4–8 Hz)",
    "Relative Band Power – Beta (13–30 Hz)",
    "Relative Band Power – Delta (0.5–4 Hz)",
    "Relative Band Power – Gamma (30–45 Hz)",
    "Theta/Beta Ratio"
]

ALL_LABEL_COLS = [
    "Stress", "Low", "Medium", "High", "Relax", "Mirror image", "Arithmetic",
    "Stroop", "Score", "HR", "HRV", "Difficulty", "Trial", "Test", "Condition",
    "Subject", "Index", "Task"
]

COMMON_EEG_CHANNELS = {
    'fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1', 'o2',
    'f7', 'f8', 't7', 't8', 'p7', 'p8', 'fz', 'cz', 'pz', 'af3', 'af4',
    'fc5', 'fc6', 'cp5', 'cp6', 'fc1', 'fc2', 'cp1', 'cp2', 'po3', 'po4',
    'ft9', 'ft10', 'po9', 'po10', 'oz', 'f1', 'f2', 'c1', 'c2', 'p1', 'p2',
    'af7', 'af8', 'fc3', 'fc4', 'cp3', 'cp4', 'po7', 'po8', 'f5', 'f6',
    'c5', 'c6', 'p5', 'p6', 't3', 't4', 't5', 't6', 'tp7', 'tp8', 'tp9', 'tp10'
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['WTF_CSRF_HEADERS'] = ['X-CSRFToken']
csrf = CSRFProtect(app)

DB_USER = os.environ.get('DB_USER') or ''
DB_PASS = os.environ.get('DB_PASS') or ''
DB_SERVER = os.environ.get('DB_SERVER') or ''
DB_NAME = os.environ.get('DB_NAME') or ''
encoded_user = quote_plus(DB_USER)
encoded_password = quote_plus(DB_PASS)
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f'mssql+pyodbc://{encoded_user}:{encoded_password}@{DB_SERVER}:1433/{DB_NAME}'
    '?driver=ODBC+Driver+17+for+SQL+Server&Connection+Timeout=60'
)
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"pool_pre_ping": True, "pool_recycle": 1800}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False
app.config['EMAIL_CONFIRM_MAX_AGE_SECONDS'] = int(os.environ.get('EMAIL_CONFIRM_MAX_AGE_SECONDS', 3600))
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() in ['true', 'on', '1']
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')

db.init_app(app)
mail.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    initialize_serializer(app)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return (img * 255).astype(np.uint8)


def _load_and_process_image(recording_id: str):
    app.logger.info(f"Starting Image data load for recording: {recording_id}")
    
    image_record = StudyRecordingImage.query.get(recording_id)
    if not image_record or not image_record.data_uri:
        return None, "ERROR: Image data URI not found in database."

    connect_str = os.environ.get('AZURE_BLOB')
    if not connect_str:
        return None, "ERROR: Azure Blob Storage connection string is not configured."

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        parsed_url = urlparse(image_record.data_uri)
        container_name = parsed_url.path.lstrip('/').split('/')[0]
        blob_name = '/'.join(parsed_url.path.lstrip('/').split('/')[1:])
        
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data_stream = blob_client.download_blob()
        blob_bytes = blob_data_stream.readall()
        
        original_filename = os.path.basename(blob_name)
        ext = os.path.splitext(original_filename)[1].lower()
        file_stream = BytesIO(blob_bytes)

        img = None
        
        # DICOM
        if ext == ".dcm":
            ds = pydicom.dcmread(file_stream)
            img_array = ds.pixel_array
            if img_array.ndim == 3:
                img_array = img_array[img_array.shape[0] // 2]
            img = normalize_to_uint8(img_array)

        # NIfTI
        elif ext in [".nii", ".gz"]:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                tmp.write(blob_bytes)
                tmp.flush()
                nii = nib.load(tmp.name)
                data = nii.get_fdata()
                img_array = data[:, :, data.shape[2] // 2]
                img_array = np.rot90(img_array)
                img = normalize_to_uint8(img_array)

        # Standard Images
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            pil_img = Image.open(file_stream).convert("L")
            img = np.array(pil_img)

        else:
            return None, f"ERROR: Unsupported image format: {ext}"

        buffer = BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64, None

    except Exception as e:
        app.logger.error(f"Error processing image {recording_id}: {e}", exc_info=True)
        return None, f"ERROR: {str(e)}"


def _load_and_detect_eeg_type(recording_id: str):

    app.logger.info(f"Starting EEG data load for recording: {recording_id}")
    eeg_record = StudyRecordingEEG.query.get(recording_id)
    if not eeg_record or not eeg_record.data_uri:
        return None, "ERROR", None, "EEG data URI not found in database."

    connect_str = os.environ.get('AZURE_BLOB')
    if not connect_str:
        return None, "ERROR", None, "Azure Blob Storage connection string is not configured."

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        parsed_url = urlparse(eeg_record.data_uri)
        container_name = parsed_url.path.lstrip('/').split('/')[0]
        blob_name = '/'.join(parsed_url.path.lstrip('/').split('/')[1:])
        
        app.logger.info(f"Downloading blob: {blob_name} from container: {container_name}")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data_stream = blob_client.download_blob()
        blob_bytes = blob_data_stream.readall()
        
        file_data = BytesIO(blob_bytes)
        original_filename = os.path.basename(blob_name)
        file_ext = os.path.splitext(original_filename)[1].lower()

        # Handle .zip files by extracting the first valid file
        if file_ext == '.zip':
            app.logger.info("Detected .zip file, attempting to extract first file...")
            with zipfile.ZipFile(file_data, 'r') as zf:
                if not zf.namelist():
                    return None, "ERROR", original_filename, "Uploaded ZIP file is empty."
                
                # Find the first non-metadata file
                first_file_name = None
                for name in zf.namelist():
                    if not name.startswith('__MACOSX') and not name.endswith('/'):
                        first_file_name = name
                        break
                
                if not first_file_name:
                    return None, "ERROR", original_filename, "Uploaded ZIP file contains no valid files."

                app.logger.info(f"Extracting '{first_file_name}' from zip...")
                with zf.open(first_file_name) as extracted_file:
                    extracted_bytes = extracted_file.read()
                    file_data = BytesIO(extracted_bytes)
                    original_filename = first_file_name # Update filename and ext for detection
                    file_ext = os.path.splitext(original_filename)[1].lower()

        # Standard Time-Domain Formats (EDF, BDF, etc.)
        if file_ext in ['.edf', '.bdf']:
            app.logger.info(f"Detected {file_ext}, loading with mne.io.read_raw_...()")
            # MNE readers need a file path, so we use a temporary file
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as tmp:
                tmp.write(file_data.getvalue())
                tmp.flush()
                if file_ext == '.edf':
                    raw = mne.io.read_raw_edf(tmp.name, preload=True)
                elif file_ext == '.bdf':
                    raw = mne.io.read_raw_bdf(tmp.name, preload=True)
            
            # Filter out non-EEG channels (metadata columns like BATTERY, CQ_*, etc.)
            all_channel_names = raw.ch_names
            eeg_channels = [ch for ch in all_channel_names if ch.lower() in COMMON_EEG_CHANNELS]
            
            if not eeg_channels:
                app.logger.error(f"No recognizable EEG channels found in {file_ext} file.")
                return None, "ERROR", original_filename, f"No recognizable EEG channels found in {file_ext} file. Found channels: {all_channel_names[:10]}..."
            
            # Pick only the EEG channels
            app.logger.info(f"Filtering from {len(all_channel_names)} channels to {len(eeg_channels)} EEG channels")
            raw.pick_channels(eeg_channels)
            
            return raw, "TIME_DOMAIN", original_filename, None

        # CSV Format (could be Time or Frequency Domain)
        elif file_ext == '.csv':
            app.logger.info("Detected .csv, reading header to determine type...")
            file_data.seek(0)
            header_line = file_data.readline().decode('utf-8').strip()
            file_data.seek(0)
            headers = [h.strip().lower() for h in header_line.split(',')]

            # Check for frequency-domain keywords
            freq_keywords = ['alpha', 'beta', 'theta', 'delta', 'gamma', 'frontal', 'central']
            if any(keyword in h for h in headers for keyword in freq_keywords):
                app.logger.info("Detected Frequency-Domain (PSD) CSV.")
                df = pd.read_csv(file_data)
                return df, "FREQUENCY_DOMAIN", original_filename, None
            
            # Assume Time-Domain CSV
            else:
                app.logger.info("Detected Time-Domain (Raw) CSV. Converting to MNE Raw object.")
                df = pd.read_csv(file_data)
                ch_names = [col for col in df.columns if col.lower() in COMMON_EEG_CHANNELS]
                if not ch_names:
                    app.logger.error(f"CSV file '{original_filename}' has no recognizable EEG channel columns.")
                    return None, "ERROR", original_filename, "CSV file has no recognizable EEG channel columns (e.g., Fp1, Fz, C3, etc.). Non-EEG columns like 'BATTERY' or 'CQ_AF3' are ignored."

                app.logger.info(f"Found {len(ch_names)} EEG channels: {ch_names}")
                
                # Data must be in Volts, MNE expects (n_channels, n_samples)
                # Let's assume data is in uV and convert to V
                data = df[ch_names].values.T * 1e-6 
                
                # Estimate sampling frequency (sfreq) if not obvious
                sfreq = 250 # Default
                time_col = None
                if 'timestamp' in headers:
                    time_col = df.columns[headers.index('timestamp')]
                elif 'time' in headers:
                    time_col = df.columns[headers.index('time')]

                if time_col:
                    try:
                        timestamps = pd.to_numeric(df[time_col])
                        # Calculate sfreq based on median diff
                        median_diff = timestamps.diff().median()
                        if median_diff > 10: # Likely milliseconds
                            sfreq = 1000.0 / median_diff
                        else: # Likely seconds
                            sfreq = 1.0 / median_diff
                        
                        if sfreq > 10000 or sfreq < 1: # Unrealistic sfreq
                            app.logger.warning(f"Unrealistic sfreq calculated ({sfreq}). Defaulting to 250 Hz.")
                            sfreq = 250
                        else:
                            app.logger.info(f"Estimated sfreq from '{time_col}' column: {sfreq:.2f} Hz")
                    except Exception:
                        app.logger.warning("Could not calculate sfreq, defaulting to 250 Hz.")
                else:
                    app.logger.info("No 'time' or 'timestamp' column found. Defaulting to 250 Hz.")

                
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                raw = mne.io.RawArray(data, info)
                try:
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage, on_missing='ignore')
                except Exception as e:
                    app.logger.warning(f"Could not set standard montage: {e}")

                return raw, "TIME_DOMAIN", original_filename, None

        # Unknown
        else:
            app.logger.error(f"Unsupported file type: {file_ext}")
            return None, "ERROR", original_filename, f"Unsupported file type ({file_ext}). Only .edf, .bdf, and .csv files are currently supported for visualization."

    except Exception as e:
        app.logger.error(f"Error in _load_and_detect_eeg_type for {recording_id}: {e}", exc_info=True)
        return None, "ERROR", None, f"An internal error occurred: {str(e)}"


def generate_synthetic_sample(s1: np.ndarray, s2: np.ndarray, noise_level: float) -> np.ndarray:
    interpolation_factor = _rng.uniform(0.3, 0.7)
    new_sample = s1 + (s2 - s1) * interpolation_factor
    noise = (
        _rng.normal(0, 1, new_sample.shape)
        * np.std(np.vstack((s1, s2)), axis=0)
        * noise_level
    )
    return new_sample + noise


def create_synthetic_dataset(features_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if len(features_df) < 2:
        if len(features_df) == 0:
            return pd.DataFrame(columns=features_df.columns)
        return pd.concat([features_df] * n_samples, ignore_index=True).head(n_samples)

    scaler = StandardScaler().fit(features_df)
    features_norm = scaler.transform(features_df)
    corr_matrix = np.corrcoef(features_norm)
    np.fill_diagonal(corr_matrix, 0)

    highly_corr_pairs = np.argwhere(corr_matrix > SYNTHESIS_CORR_THRESHOLD)

    if len(highly_corr_pairs) == 0:
        highly_corr_pairs = np.argwhere(corr_matrix > 0.5)
        if len(highly_corr_pairs) == 0:
            indices = np.arange(len(features_norm))
            pairs_i = _rng.choice(indices, n_samples)
            pairs_j = _rng.choice(indices, n_samples)
            highly_corr_pairs = np.vstack([pairs_i, pairs_j]).T

    synthetic_samples = []
    for _ in range(n_samples):
        idx1, idx2 = highly_corr_pairs[_rng.integers(len(highly_corr_pairs))]
        sample1 = features_norm[idx1]
        sample2 = features_norm[idx2]
        new_norm_sample = generate_synthetic_sample(
            sample1, sample2, SYNTHESIS_NOISE_LEVEL
        )
        synthetic_samples.append(new_norm_sample)

    synthetic_features = scaler.inverse_transform(np.array(synthetic_samples))
    return pd.DataFrame(synthetic_features, columns=features_df.columns)


def derive_eeg_biomarkers(synthetic_df: pd.DataFrame, eeg_condition: str, eeg_biomarker_map: dict) -> dict:
    """
    Derives plausible biomarker values from the synthetic PSD feature dataframe.
    This is an estimation, not a true signal processing calculation.
    """
    results = {}
    
    def get_id(name):
        if name not in eeg_biomarker_map:
            app.logger.warning(f"Demo EEG: Biomarker name '{name}' not found in database map. Skipping derivation.")
            return None
        return eeg_biomarker_map[name]

    bands = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
    band_powers = {}
    
    for band in bands:
        band_cols = [col for col in synthetic_df.columns if col.startswith(band)]
        if band_cols:
            band_powers[band] = synthetic_df[band_cols].mean().mean()
        else:
            band_powers[band] = 0.0
            
    total_power = sum(band_powers.values())
    if total_power == 0:
        total_power = 1 # Avoid division by zero

    # Calculate Relative Power
    if get_id('Relative Band Power – Alpha (8–13 Hz)'):
        results[get_id('Relative Band Power – Alpha (8–13 Hz)')] = (band_powers['Alpha'] / total_power) * 100
    if get_id('Relative Band Power – Beta (13–30 Hz)'):
        results[get_id('Relative Band Power – Beta (13–30 Hz)')] = (band_powers['Beta'] / total_power) * 100
    if get_id('Relative Band Power – Theta (4–8 Hz)'):
        results[get_id('Relative Band Power – Theta (4–8 Hz)')] = (band_powers['Theta'] / total_power) * 100
    if get_id('Relative Band Power – Delta (0.5–4 Hz)'):
        results[get_id('Relative Band Power – Delta (0.5–4 Hz)')] = (band_powers['Delta'] / total_power) * 100
    if get_id('Relative Band Power – Gamma (30–45 Hz)'):
        results[get_id('Relative Band Power – Gamma (30–45 Hz)')] = (band_powers['Gamma'] / total_power) * 100
    
    # Calculate Ratio
    if get_id('Theta/Beta Ratio'):
        if band_powers['Beta'] > 0:
            results[get_id('Theta/Beta Ratio')] = band_powers['Theta'] / band_powers['Beta']
        else:
            results[get_id('Theta/Beta Ratio')] = 0.0

    # Estimate "Peak Alpha Frequency"
    if get_id('Peak Alpha Frequency'):
        if eeg_condition in ['Relaxed', 'Normal', 'Relax Task']:
            results[get_id('Peak Alpha Frequency')] = random.uniform(9.5, 11.5)
        else: # Stressed, Arithmetic, etc.
            results[get_id('Peak Alpha Frequency')] = random.uniform(8.5, 10.0)
        
    # Clean up NaNs and Infs, ensure they are Decimal-compatible
    for key, value in results.items():
        if pd.isna(value) or not np.isfinite(value):
            results[key] = Decimal('0.0')
        else:
            # Quantize to 5 decimal places
            results[key] = Decimal(value).quantize(Decimal('0.00001'))

    return results


def generate_demo_eeg_and_biomarkers(
    subject: Subject,
    study_id_str: str,
    eeg_condition: str,
    blob_service_client: BlobServiceClient,
    biomarker_types_in_study: set,
    eeg_biomarker_map: dict
):
    try:
        app.logger.info(f"Demo EEG: Starting generation for Subject {subject.external_subject_code} (Condition: {eeg_condition})")

        if eeg_condition not in CONDITION_MAP:
            app.logger.warning(f"Demo EEG: Unknown condition '{eeg_condition}'. Skipping.")
            return

        blob_name, label_col, label_val = CONDITION_MAP[eeg_condition]
        source_container_client = blob_service_client.get_container_client("eeg")
        try:
            blob_client = source_container_client.get_blob_client(blob_name)
            downloader = blob_client.download_blob()
            blob_bytes = downloader.readall()
            df = pd.read_csv(BytesIO(blob_bytes))
        except ResourceNotFoundError:
            app.logger.error(f"Demo EEG: Source file '{blob_name}' not found in 'eeg' container. Skipping.")
            return
        except Exception as e:
            app.logger.error(f"Demo EEG: Failed to download or read source blob '{blob_name}'. {e}")
            return

        # Generate Synthetic Data (6 epochs = 60 seconds)
        duration_s = 60
        n_samples = max(1, int(round(duration_s / EPOCH_DURATION_S)))
        
        condition_df = df[df[label_col] == label_val]
        if condition_df.empty:
            app.logger.warning(f"Demo EEG: No source data found for condition '{eeg_condition}'. Skipping.")
            return

        source_features_df = condition_df.drop(columns=[col for col in ALL_LABEL_COLS if col in condition_df.columns], errors='ignore')
        synthetic_df = create_synthetic_dataset(source_features_df, n_samples)
        
        if synthetic_df.empty:
            app.logger.warning("Demo EEG: Synthetic dataframe was empty. Skipping.")
            return

        # Save Synthetic EEG CSV to 'recordings' container
        csv_bytes = synthetic_df.to_csv(index=False).encode('utf-8')
        subject_id_str = str(subject.subject_id)
        # Store as .csv, but mimic the .zip convention from the main app
        recording_blob_name = f"{study_id_str}/{subject_id_str}/{str(uuid.uuid4())}.csv" 
        
        dest_container_client = blob_service_client.get_container_client("recordings")
        dest_blob_client = dest_container_client.get_blob_client(recording_blob_name)
        dest_blob_client.upload_blob(csv_bytes, overwrite=True)
        data_uri = dest_blob_client.url

        # Create DB Records for the EEG file
        rec_datetime = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=random.randint(9, 16))
        
        new_eeg_rec = StudyRecording(
            study_id=study_id_str,
            subject_id=subject_id_str,
            recording_datetime=rec_datetime,
            recording_type='EEG'
        )
        db.session.add(new_eeg_rec)
        db.session.flush() # Need the recording_id

        new_eeg_data = StudyRecordingEEG(
            recording=new_eeg_rec,
            study_id=study_id_str,
            subject_id=subject_id_str,
            data_uri=data_uri,
            eeg_id=None # No specific device
        )
        db.session.add(new_eeg_data)
        
        # Derive and Save Biomarkers
        derived_biomarkers = derive_eeg_biomarkers(
            synthetic_df, 
            eeg_condition,
            eeg_biomarker_map
        )
        
        for biomarker_id, value in derived_biomarkers.items():
            if biomarker_id in biomarker_types_in_study:
                new_bio_rec = StudyRecording(
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    recording_datetime=rec_datetime,
                    recording_type='Biomarker'
                )
                db.session.add(new_bio_rec)
                db.session.flush()
                
                new_biomarker_record = StudyRecordingBiomarker(
                    recording=new_bio_rec,
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    biomarker_id=biomarker_id,
                    biomarker_value=value
                )
                db.session.add(new_biomarker_record)
        
        app.logger.info(f"Demo EEG: Queued EEG file and {len(derived_biomarkers)} derived biomarkers for {subject.external_subject_code}")

    except Exception as e:
        app.logger.error(f"Demo EEG: FAILED for Subject {subject.external_subject_code}. Error: {e}", exc_info=True)
        raise e


def _format_variable_name(name):
    name = str(name)
    prefixes = {
        "bio_": "Bio: ",
        "sym_": "Symptom: ",
        "ae_": "AE: ",
        "med_": "Med: ",
        "arm_name_": "Arm: ",
        "gender_": "Gender: ",
        "ethnicity_": "Ethnicity: ",
        "race_": "Race: "
    }
    
    for prefix, replacement in prefixes.items():
        if name.startswith(prefix):
            # Replace remaining underscores and handle 'nan' representation
            processed_name = name[len(prefix):].replace("_", " ").replace(" nan", " (N/A)")
            return replacement + processed_name
    
    # For simple names like 'age' or 'height_cm'
    processed_name = name.replace("_", " ").replace(" nan", " (N/A)")
    # Capitalize first letter only
    return processed_name[0].upper() + processed_name[1:] if processed_name else ""


_ALLOWED_SUBJECT_FIELDS = frozenset([
    'gender', 'ethnicity', 'race', 'handedness', 'smoking_status',
    'alcohol_intake', 'first_name', 'last_name', 'status',
    'height_cm', 'weight_kg',
])


_ALLOWED_UPLOAD_EXTENSIONS = frozenset([
    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'txt',
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff',
    'zip', 'edf', 'nii', 'dcm',
])
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

_failed_logins: dict = defaultdict(list)
_failed_logins_lock = threading.Lock()
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 300


def _login_blocked(ip: str) -> bool:
    """Return True if this IP has exceeded the failed-login threshold."""
    if app.config.get('TESTING'):
        return False
    now = datetime.now(timezone.utc)
    with _failed_logins_lock:
        recent = [t for t in _failed_logins[ip]
                  if (now - t).total_seconds() < _LOGIN_WINDOW_SECONDS]
        _failed_logins[ip] = recent
        return len(recent) >= _LOGIN_MAX_ATTEMPTS


def _record_failed_login(ip: str) -> None:
    if app.config.get('TESTING'):
        return
    with _failed_logins_lock:
        _failed_logins[ip].append(datetime.now(timezone.utc))


def _allowed_upload(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in _ALLOWED_UPLOAD_EXTENSIONS


def _require_study_membership(study_id):
    """Abort with 403 if the current user is not a participant in the study."""
    from flask import abort
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    if not participant:
        abort(403)
    link = StudyParticipantLink.query.filter_by(
        study_id=study_id,
        participant_id=str(participant.participant_id).lower(),
    ).first()
    if not link:
        abort(403)
    return link


def _build_demographic_table(count_pivot, percent_pivot):
    """Builds an HTML table for the demographic distribution report."""
    html = "<div class='table-responsive mb-3'>"
    html += "<table class='table table-sm table-striped table-bordered'>"
    html += "<thead class='thead-light'><tr><th>Category</th>"

    for arm_name in count_pivot.columns:
        html += f"<th colspan='2' class='text-center'>{html_escape(str(arm_name))}</th>"
    html += "</tr><tr><th></th>"

    for _ in count_pivot.columns:
        html += "<th class='text-center'>N</th><th class='text-center'>%</th>"
    html += "</tr></thead><tbody>"

    for category, row in count_pivot.iterrows():
        is_total_row = (category == 'Total')
        row_style = "fw-bold bg-light" if is_total_row else ""
        html += f"<tr class='{row_style}'><td>{html_escape(str(category))}</td>"
        
        for arm_name in count_pivot.columns:
            count = row[arm_name]
            percent = percent_pivot.loc[category, arm_name]
            html += f"<td class='text-center'>{count}</td>"
            html += f"<td class='text-center'>{percent:.1f}%</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


def _build_distribution_table(count_pivot, percent_pivot, category_label, arm_names):
    """Builds an HTML table for distribution reports (Diagnoses, Meds)."""
    html = "<div class='table-responsive mb-3'>"
    html += "<table class='table table-sm table-striped table-bordered'>"
    html += "<thead class='thead-light'><tr><th>Category</th>"
    
    all_cols = list(arm_names) + ['Total']
    
    for col_name in all_cols:
        html += f"<th colspan='2' class='text-center'>{html_escape(str(col_name))}</th>"
    html += "</tr><tr><th>" + html_escape(str(category_label)) + "</th>"

    for _ in all_cols:
        html += "<th class='text-center'>N</th><th class='text-center'>%</th>"
    html += "</tr></thead><tbody>"

    for category, row in count_pivot.iterrows():
        is_total_row = (category == 'Total Subjects (N)')
        row_style = "fw-bold bg-light" if is_total_row else ""
        html += f"<tr class='{row_style}'><td>{html_escape(str(category))}</td>"
        
        for col_name in all_cols:
            count = int(row[col_name])
            percent = percent_pivot.loc[category, col_name]
            percent_str = f"{percent:.1f}%" if not is_total_row else ""
            
            html += f"<td class='text-center'>{count}</td>"
            html += f"<td class='text-center'>{percent_str}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


@app.route('/study/<study_id>/download_template', methods=['GET'])
@login_required
def download_biomarker_template(study_id):
    try:
        study = Study.query.get_or_404(study_id)
        study_settings = study.settings
        
        if not study_settings:
            flash('This study has no associated settings.', 'danger')
            return redirect(url_for('edit_study', study_id=study_id))

        allowed_biomarkers = study_settings.allowed_biomarkers
        value_based_biomarkers = [
            b for b in allowed_biomarkers 
            if b.sample_type.lower() not in ['eeg', 'wearable']
        ]
        value_based_biomarkers.sort(key=lambda x: x.biomarker_name)

        if not value_based_biomarkers:
            flash('No value-based biomarkers (e.g., Blood, Saliva, Scale) are enabled for this study. Cannot generate template.', 'warning')
            return redirect(url_for('edit_study', study_id=study_id))

        base_columns = ['Subject ID', 'Date', 'Time']
        biomarker_columns = [b.biomarker_name for b in value_based_biomarkers]
        all_columns = base_columns + biomarker_columns
        df = pd.DataFrame(columns=all_columns)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data_Capture', index=False)
            for i, col in enumerate(all_columns):
                worksheet = writer.sheets['Data_Capture']
                worksheet.set_column(i, i, max(len(col), 15))
            instructions_ws = writer.book.add_worksheet('Instructions')
            instructions_ws.write_string('A1', 'Instructions:')
            instructions_ws.write_string('A2', '1. Enter Subject ID (must match the ID in the system).')
            instructions_ws.write_string('A3', '2. Enter Date in YYYY-MM-DD format.')
            instructions_ws.write_string('A4', '3. Enter Time in HH:MM format (24-hour).')
            instructions_ws.write_string('A5', '4. Fill in the biomarker values for each subject visit.')
            instructions_ws.set_column(0, 0, 70)

        output.seek(0)
        filename = f"{study.name.replace(' ', '_')}_Biomarker_Template.xlsx"
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        app.logger.error(f"Error generating biomarker template for study {study_id}: {e}", exc_info=True)
        flash(f'An error occurred while generating the template: {e}', 'danger')
        return redirect(url_for('edit_study', study_id=study_id))


@app.route('/study/<study_id>/import_template', methods=['POST'])
@login_required
def import_biomarker_template(study_id):
    _require_study_membership(study_id)
    if 'template_file' not in request.files:
        flash('No file selected for upload.', 'danger')
        return redirect(url_for('study_recordings', study_id=study_id))

    file = request.files['template_file']
    if not file or not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        flash('Invalid file. Please upload the .xlsx template.', 'danger')
        return redirect(url_for('study_recordings', study_id=study_id))

    try:
        df = pd.read_excel(file)
        required_cols = ['Subject ID', 'Date', 'Time']
        if not all(col in df.columns for col in required_cols):
            flash(f'Template is missing required columns. Must contain: {", ".join(required_cols)}', 'danger')
            return redirect(url_for('study_recordings', study_id=study_id))
        study_settings = StudySettings.query.get(study_id)
        if not study_settings:
            flash('Cannot import: Study settings not found.', 'danger')
            return redirect(url_for('study_recordings', study_id=study_id))

        value_biomarkers = [
            b for b in study_settings.allowed_biomarkers 
            if b.sample_type.lower() not in ['eeg', 'wearable']
        ]
        biomarker_map = {b.biomarker_name: b.biomarker_id for b in value_biomarkers}
        subjects = Subject.query.filter_by(study_id=study_id).all()
        subject_map = {s.external_subject_code: s.subject_id for s in subjects}
        biomarker_cols_in_file = [
            col for col in df.columns 
            if col not in required_cols and col in biomarker_map
        ]
        unknown_cols = [
            col for col in df.columns 
            if col not in required_cols and col not in biomarker_map
        ]
        if not biomarker_cols_in_file:
            flash('No valid biomarker columns found in the file that match this study\'s settings.', 'danger')
            return redirect(url_for('study_recordings', study_id=study_id))

        inserted_count = 0
        updated_count = 0
        skipped_rows = 0
        skipped_cells_value = 0
        g.disable_auditing = True

        for index, row in df.iterrows():
            subject_code = str(row['Subject ID']).strip()
            date_str = str(row['Date']).split(' ')[0]
            time_str = str(row['Time'])
            if subject_code not in subject_map:
                app.logger.warning(f"Import Skip (Row {index+2}): Subject ID '{subject_code}' not found in study.")
                skipped_rows += 1
                continue
                
            subject_id = subject_map[subject_code]

            try:
                parsed_time = pd.to_datetime(time_str, errors='coerce').time()
                if pd.isna(parsed_time):
                    parsed_time = datetime.min.time()
                parsed_date = pd.to_datetime(date_str, errors='raise').date()
                recording_datetime = datetime.combine(parsed_date, parsed_time)
                
            except Exception as e:
                app.logger.warning(f"Import Skip (Row {index+2}): Invalid Date '{date_str}'. Error: {e}")
                skipped_rows += 1
                continue

            for biomarker_name in biomarker_cols_in_file:
                value = row[biomarker_name]
                if pd.isna(value):
                    continue

                try:
                    biomarker_id = biomarker_map[biomarker_name]
                    biomarker_value_decimal = Decimal(str(value))
                except (InvalidOperation, ValueError, TypeError) as e:
                    app.logger.warning(f"Import Skip (Row {index+2}, Col '{biomarker_name}'): Invalid value '{value}'. Error: {e}")
                    skipped_cells_value += 1
                    continue

                existing_entry = db.session.query(StudyRecordingBiomarker)\
                    .join(StudyRecording, StudyRecording.recording_id == StudyRecordingBiomarker.recording_id)\
                    .filter(
                        StudyRecording.subject_id == subject_id,
                        StudyRecording.recording_datetime == recording_datetime,
                        StudyRecordingBiomarker.biomarker_id == biomarker_id
                    ).first()

                if existing_entry:
                    if existing_entry.biomarker_value != biomarker_value_decimal:
                        existing_entry.biomarker_value = biomarker_value_decimal
                        updated_count += 1
                else:
                    new_rec_parent = StudyRecording(
                        study_id=study_id,
                        subject_id=subject_id,
                        recording_datetime=recording_datetime,
                        recording_type='Biomarker'
                    )
                    biomarker_type_obj = BiomarkerType.query.get(biomarker_id)
                    if biomarker_type_obj and biomarker_type_obj.sample_type.lower() == 'scale':
                        new_rec_parent.recording_type = 'Scale'
                        
                    db.session.add(new_rec_parent)
                    new_biomarker_rec = StudyRecordingBiomarker(
                        recording=new_rec_parent,
                        study_id=study_id,
                        subject_id=subject_id,
                        biomarker_id=biomarker_id,
                        biomarker_value=biomarker_value_decimal
                    )
                    db.session.add(new_biomarker_rec)
                    inserted_count += 1
        
        db.session.commit()
        
        success_message = f"Import successful: {inserted_count} new records added, {updated_count} records updated."
        if skipped_rows > 0:
            success_message += f" {skipped_rows} rows were skipped (invalid Subject ID or Date)."
        if skipped_cells_value > 0:
            success_message += f" {skipped_cells_value} cells were skipped (invalid value)."
        if unknown_cols:
            success_message += f" Ignored columns not in study settings: {', '.join(unknown_cols)}."

        flash(success_message, 'success')

    except pd.errors.EmptyDataError:
        flash('The uploaded file is empty.', 'warning')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error importing biomarker template for study {study_id}: {e}", exc_info=True)
        flash(f'An error occurred during import: {e}', 'danger')
    finally:
        g.disable_auditing = False
    return redirect(url_for('study_recordings', study_id=study_id))


@app.route('/visualize_eeg/<recording_id>', methods=['GET'])
@login_required
def visualize_eeg(recording_id):
    """
    Renders the EEG visualization page.
    This route detects the data type and passes it to the template,
    which then shows the appropriate visualization options.
    """
    recording = StudyRecording.query.get_or_404(recording_id)
    subject = Subject.query.get_or_404(recording.subject_id)
    study = Study.query.get_or_404(str(recording.study_id))
    
    # Check for authorization
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    if not participant:
         flash('You must be a participant in this study to view recordings.', 'danger')
         return redirect(url_for('main_app'))
    
    link = StudyParticipantLink.query.filter_by(
        study_id=study.study_id, 
        participant_id=participant.participant_id
    ).first()
    
    if not link:
        flash('You are not a participant in this study.', 'danger')
        return redirect(url_for('main_app'))

    # Load and detect data type
    _, data_type, _, error_message = _load_and_detect_eeg_type(recording_id)

    if data_type == "ERROR":
        flash(f"Could not visualize EEG: {error_message}", 'danger')
        return redirect(url_for('study_recordings', study_id=study.study_id))

    data_type_display = "Unknown"
    if data_type == "TIME_DOMAIN":
        data_type_display = "Time-Domain"
    elif data_type == "FREQUENCY_DOMAIN":
        data_type_display = "Frequency-Domain (PSD)"

    return render_template(
        'visualize_eeg.html',
        study=study,
        subject=subject,
        recording=recording,
        data_type=data_type,
        data_type_display=data_type_display,
        error_message=error_message
    )


@app.route('/api/visualize_eeg/<recording_id>/<plot_type>', methods=['GET'])
@login_required
def api_visualize_eeg_plot(recording_id, plot_type):
    """
    API endpoint that generates and returns the HTML for a specific MNE plot
    or a Matplotlib plot.
    """
    try:
        data_object, data_type, _, error_message = _load_and_detect_eeg_type(recording_id)

        if data_type == "ERROR":
            return jsonify({'error': error_message}), 400

        plot_html = f'<div class="alert alert-warning">Plot type "{plot_type}" is not yet implemented for {data_type}.</div>'

        if data_type == "TIME_DOMAIN":
            raw = data_object
            
            # Filter to only keep EEG channels
            all_channel_names = raw.ch_names
            eeg_channels = [ch for ch in all_channel_names if ch.lower() in COMMON_EEG_CHANNELS]
            
            if not eeg_channels:
                return jsonify({'error': 'No recognizable EEG channels found in the data.'}), 400
            
            # Pick only the EEG channels
            if len(eeg_channels) < len(all_channel_names):
                app.logger.info(f"Filtering to {len(eeg_channels)} EEG channels for plotting")
                raw = raw.copy().pick_channels(eeg_channels)
            
            # Set a standard montage for plotting
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')
            except Exception as e:
                app.logger.warning(f"Could not set standard montage for {recording_id}: {e}")

            if plot_type == 'psd_plot':
                psd = raw.compute_psd(fmax=50)
                fig = psd.plot(average=False, show=False)
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                plot_html = f'<img src="data:image/png;base64,{img_data}" alt="PSD Plot" class="img-fluid" style="display: block; margin: 0 auto; max-width: 100%;">'
                plt.close(fig)

            elif plot_type == 'topo_maps':
                psd = raw.compute_psd(fmax=50)
                fig = psd.plot_topomap(bands={'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
                                              'Beta': (13, 30), 'Gamma': (30, 50)}, 
                                      show=False)
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                plot_html = f'<img src="data:image/png;base64,{img_data}" alt="Topographic Maps" class="img-fluid" style="display: block; margin: 0 auto; max-width: 100%;">'
                plt.close(fig)

        elif data_type == "FREQUENCY_DOMAIN":
            df = data_object
            
            if plot_type == 'topo_band':
                # This data is regional, not channel-based. We cannot draw a topoplot.
                plot_html = """
                <div class="alert alert-warning">
                    <h4 class="alert-heading">Visualization Not Available for this Data</h4>
                    <p>Topographic plots require data from individual channels (e.g., Fp1, Fz, Pz) to map power to specific scalp locations.</p>
                    <hr>
                    <p class="mb-0">
                        This EEG file contains <strong>regionally-aggregated data</strong> (e.g., "Frontal", "Central"). 
                        This is great for bar charts, but doesn't have the spatial detail for a topoplot.
                        <br>
                        Please choose the <strong>"Regional Power Bar Chart"</strong> option instead, or upload a raw (time-domain) EEG file to see topographic maps.
                    </p>
                </div>
                """
            
            elif plot_type == 'bar_regional':
                bands = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
                band_cols = [col for col in df.columns if any(b in col for b in bands) and any(r in col for r in ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal'])]
                if not band_cols:
                    return jsonify({'error': 'No regionally-aggregated PSD columns (e.g., "Alpha Frontal") found in this file.'}), 400

                file_mean_psd = df[band_cols].mean()
                
                band_data = {band: [] for band in bands}
                regions = []
                
                for col_name in band_cols:
                    for r in ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal']:
                        if r in col_name and r not in regions:
                            regions.append(r)
                            
                for band in bands:
                    for region in regions:
                        col_name = f"{band} {region}" # Assumes "Band Region" format
                        if col_name in file_mean_psd and pd.notna(file_mean_psd[col_name]):
                            band_data[band].append(file_mean_psd[col_name])
                        else:
                            band_data[band].append(0) 

                data = {'regions': regions}
                data.update(band_data)
                plot_df = pd.DataFrame(data).set_index('regions')
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_df.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title("Mean PSD Power by Region and Band", fontsize=16)
                ax.set_ylabel("Mean PSD Power (μV²/Hz)")
                ax.set_xlabel("Brain Region")
                ax.legend(
                    loc='upper left', 
                    bbox_to_anchor=(1.02, 1),  # push legend outside the axes
                    ncol=1,                    # vertical legend
                    fancybox=True, 
                    shadow=False
                )
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=0)
                plt.tight_layout(rect=[0, 0.03, 1, 0.93])

                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                plot_html = f'<img src="data:image/png;base64,{img_data}" alt="Regional Power Bar Chart" class="img-fluid" style="display: block; margin: 0 auto;">'
                plt.close(fig)

        return jsonify({'plot_html': plot_html})

    except Exception as e:
        app.logger.error(f"Error generating EEG plot {plot_type} for {recording_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.before_request
def store_user_in_g():
    """
    Stores the current user's email in the Flask 'g' object.
    This makes it accessible to SQLAlchemy event listeners within the same request context.
    """
    if current_user and current_user.is_authenticated:
        g.user_email = current_user.email
    else:
        g.user_email = 'anonymous'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/learn_more', methods=['GET'])
def learn_more():
    """
    Renders the 'Learn More' page which contains detailed information,
    a brochure download, and the pricing calculator.
    """
    return render_template('learn_more.html')


@app.route('/best_practices', methods=['GET'])
@login_required
def best_practices():
    """Renders the Best Practices Guide page."""
    return render_template('best_practices.html')


@app.route('/help', methods=['GET'])
@login_required
def help_content():
    page = request.args.get('page', 'default')
    return jsonify(HELP_CONTENT.get(page, HELP_CONTENT['default']))


@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('index'))


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/faq', methods=['GET'])
def faq():
    return render_template('faq.html')


@app.route('/features', methods=['GET'])
def features():
    return render_template('features.html')


def send_confirmation_email(user_email):
    """Sends the confirmation email."""
    token = extensions.s.dumps(user_email, salt='email-confirm-salt')
    confirm_url = url_for('confirm_email', token=token, _external=True)
    msg = Message('Confirm Your Email for STUDYPRO',
                  recipients=[user_email],
                  html=render_template('email/activate.html', confirm_url=confirm_url))
    mail.send(msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main_app'))
    
    is_local = '127.0.0.1' in request.host or 'localhost' in request.host

    if request.method == 'POST':
        if not is_local:
            registration_code = request.form['registration_code']
            beta_key = os.environ.get('BETA_KEY')

            if not beta_key or registration_code != beta_key:
                flash('Please contact us for a registration code prior to registration...', 'danger')
                return redirect(url_for('register'))

        email = request.form['email']
        password = request.form['password']
        password_confirm = request.form['password_confirm']

        if password != password_confirm:
            flash('Passwords do not match. Please try again.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first() is not None:
            flash('Email address already registered.', 'warning')
            return redirect(url_for('register'))
        
        if is_local:
            new_user = User(email=email, email_confirmed=True)
        else:
            new_user = User(email=email)

        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        if is_local:
            flash('Local account created successfully. You may now log in.', 'success')
        else:
            send_confirmation_email(new_user.email)
            flash('A confirmation email has been sent. Please check your inbox to activate your account.', 'info')
            
        return redirect(url_for('login'))
        
    return render_template('register.html', is_local=is_local)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main_app'))
        
    if request.method == 'POST':
        ip = request.remote_addr or '0.0.0.0'
        if _login_blocked(ip):
            flash('Too many failed login attempts. Please try again later.', 'danger')
            return redirect(url_for('login'))

        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            if user.email_confirmed:
                login_user(user)
                return redirect(url_for('main_app'))
            else:
                flash('Please confirm your email address before logging in.', 'warning')
                return redirect(url_for('login'))
        else:
            _record_failed_login(ip)
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))
            
    return render_template('login.html')


@app.route('/confirm_email/<token>', methods=['GET'])
def confirm_email(token):
    try:
        email = extensions.s.loads(
            token, salt='email-confirm-salt',
            max_age=app.config.get('EMAIL_CONFIRM_MAX_AGE_SECONDS', 3600)
        )
    except BadSignature:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    user = User.query.filter_by(email=email).first_or_404()
    if user.email_confirmed:
        flash('Account already confirmed. Please login.', 'info')
    else:
        user.email_confirmed = True
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
    return redirect(url_for('login'))


@app.route('/main', methods=['GET'])
@login_required
def main_app():
    user_as_participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    studies_data = []

    is_admin = False
    if user_as_participant:
        admin_link_exists = StudyParticipantLink.query.filter_by(
            participant_id=user_as_participant.participant_id,
            is_admin=True
        ).first()
        if admin_link_exists:
            is_admin = True

    if user_as_participant:
        participant_id = str(user_as_participant.participant_id)
        user_studies_links = StudyParticipantLink.query.filter_by(participant_id=participant_id).all()
        
        if user_studies_links:
            study_ids = [link.study_id for link in user_studies_links]

            subject_count_subquery = db.session.query(
                Subject.study_id,
                func.count(Subject.subject_id).label('subject_count')
            ).group_by(Subject.study_id).subquery()

            studies_query = db.session.query(
                Study,
                subject_count_subquery.c.subject_count
            ).outerjoin(
                subject_count_subquery, Study.study_id == subject_count_subquery.c.study_id
            ).filter(Study.study_id.in_(study_ids)).order_by(Study.created_at.desc()).all()

            studies_data = [
                {
                    "study_id": study.study_id,
                    "name": study.name,
                    "description": study.description,
                    "status": study.status,
                    "start_date": study.start_date.strftime('%Y-%m-%d') if study.start_date else None,
                    "subject_count": subject_count if subject_count is not None else 0
                } for study, subject_count in studies_query
            ]

    return render_template('main.html', studies_data=studies_data, is_admin=is_admin)


@app.route('/create_study/step1', methods=['GET', 'POST'])
@login_required
def create_study_step1():
    """First step of the study creation wizard: selecting a study type and settings."""
    session.pop('new_study_type_id', None)
    session.pop('new_study_arms', None)
    session.pop('new_study_participants', None)
    session.pop('new_study_details', None)
    session.pop('new_study_settings', None)

    if request.method == 'POST':
        study_type_id = request.form.get('study_type')
        
        if not study_type_id:
            flash('Please select a study type.', 'warning')
            study_types = StudyType.query.order_by(StudyType.category, StudyType.study_type).all()
            all_biomarkers = BiomarkerType.query.order_by(BiomarkerType.sample_type, BiomarkerType.biomarker_name).all()
            return render_template('create_study_step1.html', 
                                   study_types=study_types, 
                                   all_biomarkers=all_biomarkers)
            
        session['new_study_type_id'] = study_type_id
        study_settings_data = {
            'ai_enabled': request.form.get('ai_enabled') == 'true',
            'ai_api_key': request.form.get('ai_api_key') or None,
            'eeg_enabled': request.form.get('eeg_enabled') == 'true',
            'wearables_enabled': request.form.get('wearables_enabled') == 'true',
            'biological_enabled': request.form.get('biological_enabled') == 'true',
            'scales_enabled': request.form.get('scales_enabled') == 'true',
            'biomarker_ids': request.form.getlist('biomarker_types')
        }
        
        if not study_settings_data['ai_enabled']:
            study_settings_data['ai_api_key'] = None
            
        session['new_study_settings'] = study_settings_data
        return redirect(url_for('create_study_step2'))

    study_types = StudyType.query.order_by(StudyType.category, StudyType.study_type).all()
    all_biomarkers = BiomarkerType.query.order_by(BiomarkerType.sample_type, BiomarkerType.biomarker_name).all()
    
    return render_template('create_study_step1.html', 
                           study_types=study_types, 
                           all_biomarkers=all_biomarkers)


@app.route('/create_study/step2', methods=['GET', 'POST'])
@login_required
def create_study_step2():
    """Second step: Define study arms using a server-side approach."""
    study_type_id = session.get('new_study_type_id')
    if not study_type_id:
        flash('Please complete Step 1 first.', 'warning')
        return redirect(url_for('create_study_step1'))

    if request.method == 'POST':
        action = request.form.get('action')
        
        arms_data = []
        i = 0
        while f'arm_name_{i}' in request.form:
            if action != f'remove_{i}':
                arm_name = request.form.get(f'arm_name_{i}', '')
                description = request.form.get(f'description_{i}', '')
                arms_data.append({'arm_name': arm_name, 'description': description})
            i += 1
        
        if action == 'add_arm':
            arms_data.append({'arm_name': '', 'description': ''})
            session['new_study_arms'] = arms_data
            return render_template('create_study_step2.html', arms_data=arms_data)
        
        elif action and action.startswith('remove_'):
            session['new_study_arms'] = arms_data
            return render_template('create_study_step2.html', arms_data=arms_data)
        
        else:
            if not arms_data:
                flash('Please define at least one study arm.', 'warning')
                session['new_study_arms'] = arms_data
                return render_template('create_study_step2.html', arms_data=arms_data)
            
            for i, arm in enumerate(arms_data):
                arm_name = arm.get('arm_name', '').strip()
                description = arm.get('description', '').strip()
                if not arm_name or not description:
                    flash(f'Row {i+1} is incomplete. Please provide both an arm name and a description for every entry.', 'warning')
                    session['new_study_arms'] = arms_data
                    return render_template('create_study_step2.html', arms_data=arms_data)

            seen_names = set()
            seen_descriptions = set()
            for group in arms_data:
                current_arm_name = group.get('arm_name', '').strip()
                description = group.get('description', '').strip()

                if current_arm_name in seen_names:
                    flash(f'Duplicate arm name found: "{current_arm_name}". Please ensure all arm names are unique.', 'warning')
                    session['new_study_arms'] = arms_data
                    return render_template('create_study_step2.html', arms_data=arms_data)
                seen_names.add(current_arm_name)

                if description in seen_descriptions:
                    flash(f'Duplicate description found: "{description}". Please ensure all descriptions are unique.', 'warning')
                    session['new_study_arms'] = arms_data
                    return render_template('create_study_step2.html', arms_data=arms_data)
                seen_descriptions.add(description)
            
            session['new_study_arms'] = arms_data
            return redirect(url_for('create_study_step3'))

    arms_data = session.get('new_study_arms')
    if arms_data is None:
        arms_data = [{'arm_name': '', 'description': ''}]
        session['new_study_arms'] = arms_data

    return render_template('create_study_step2.html', arms_data=arms_data)


@app.route('/create_study/step3', methods=['GET', 'POST'])
@login_required
def create_study_step3():
    if 'new_study_arms' not in session:
        flash('Please complete Step 2 first.', 'warning')
        return redirect(url_for('create_study_step2'))
    
    if 'new_study_settings' not in session:
        flash('Please complete Step 1 first.', 'warning')
        return redirect(url_for('create_study_step1'))

    if 'new_study_participants' not in session:
        session['new_study_participants'] = []
    if 'new_study_details' not in session:
        session['new_study_details'] = {}

    if request.method == 'POST':
        if 'study_name' in request.form and 'study_description' in request.form:
            session['new_study_details'] = {
                'name': request.form.get('study_name', ''),
                'description': request.form.get('study_description', ''),
                'start_date': request.form.get('start_date', ''),
                'end_date': request.form.get('end_date', ''),
            }
            session.modified = True

        action = request.form.get('action')
        participants = session.get('new_study_participants', [])

        if action == 'save_participant':
            participant_data = {
                'temp_id': request.form.get('temp_id') or str(uuid.uuid4()),
                'first_name': request.form.get('first_name', '').strip(),
                'last_name': request.form.get('last_name', '').strip(),
                'email': request.form.get('email', '').strip(),
                'phone': request.form.get('phone', '').strip(),
                'role': request.form.get('role', '').strip(),
                'affiliation': request.form.get('affiliation', '').strip(),
                'department': request.form.get('department', '').strip(),
                'orcid': request.form.get('orcid', '').strip(),
                'country': request.form.get('country', '').strip(),
                'funding_role': request.form.get('funding_role', '').strip(),
                'percent_effort': request.form.get('percent_effort'),
                'salary_contribution': request.form.get('salary_contribution'),
            }
            p_index = next((i for i, p in enumerate(participants) if p.get('temp_id') == participant_data['temp_id']), None)
            
            if p_index is not None:
                participants[p_index] = participant_data
            else:
                participants.append(participant_data)
            session['new_study_participants'] = participants
            return redirect(url_for('create_study_step3'))

        elif action == 'remove_participant':
            participant_id_to_remove = request.form.get('temp_id')
            participants = [p for p in participants if p.get('temp_id') != participant_id_to_remove]
            session['new_study_participants'] = participants
            return redirect(url_for('create_study_step3'))

        elif action == 'save_study':
            study_details = session.get('new_study_details', {})
            study_name = study_details.get('name', '').strip()
            study_description = study_details.get('description', '').strip()
            start_date_str = study_details.get('start_date', '').strip()
            end_date_str = study_details.get('end_date', '').strip()

            if not study_name or not study_description:
                flash('Study Name and Description are required.', 'warning')
                return redirect(url_for('create_study_step3'))
            if not participants:
                flash('You must add at least one participant.', 'warning')
                return redirect(url_for('create_study_step3'))
            if not any(p.get('role', '').lower() == 'principal investigator' for p in participants):
                flash('At least one participant must be assigned the role "Principal Investigator".', 'warning')
                return redirect(url_for('create_study_step3'))

            current_user_in_list = any(p.get('email', '').lower() == current_user.email.lower() for p in participants)
            if not current_user_in_list:
                flash('You must add yourself as a participant to the study in any role.', 'warning')
                return redirect(url_for('create_study_step3'))

            from datetime import date as date_type
            def _parse_date(s):
                try:
                    return date_type.fromisoformat(s) if s else None
                except ValueError:
                    return None

            try:
                new_study = Study(
                    name=study_name,
                    description=study_description,
                    study_type_id=session['new_study_type_id'],
                    status='Planned',
                    start_date=_parse_date(start_date_str),
                    end_date=_parse_date(end_date_str),
                    created_at=datetime.now(timezone.utc)
                )
                db.session.add(new_study)
                db.session.flush() 

                study_settings_data = session.get('new_study_settings')
                if study_settings_data:
                    new_settings = StudySettings(
                        study_id=new_study.study_id,
                        ai_enabled=study_settings_data['ai_enabled'],
                        ai_api_key=study_settings_data['ai_api_key'],
                        eeg_enabled=study_settings_data['eeg_enabled'],
                        wearables_enabled=study_settings_data['wearables_enabled'],
                        biological_enabled=study_settings_data['biological_enabled'],
                        scales_enabled=study_settings_data['scales_enabled'] # <-- ADDED
                    )
                    
                    selected_ids = study_settings_data.get('biomarker_ids', [])
                    if selected_ids:
                        selected_biomarkers = BiomarkerType.query.filter(
                            BiomarkerType.biomarker_id.in_(selected_ids)
                        ).all()
                        new_settings.allowed_biomarkers = selected_biomarkers
                    
                    db.session.add(new_settings)

                for group_data in session.get('new_study_arms', []):
                    new_arm = StudyArm(
                        study_id=str(new_study.study_id),
                        arm_name=group_data['arm_name'],
                        description=group_data['description']
                    )
                    db.session.add(new_arm)
                
                for p_data in participants:
                    participant_email = p_data.get('email', '').strip().lower()
                    participant_id_to_link = None

                    participant = StudyParticipant.query.filter_by(email=participant_email).first()

                    if participant:
                        participant.first_name = p_data['first_name']
                        participant.last_name = p_data['last_name']
                        participant.phone = p_data['phone']
                        participant.role = p_data['role']
                        participant.affiliation = p_data['affiliation']
                        participant.department = p_data['department']
                        participant.orcid = p_data['orcid']
                        participant.country = p_data['country']
                        participant.funding_role = p_data['funding_role']
                        participant.percent_effort = p_data.get('percent_effort') or None
                        participant.salary_contribution = p_data.get('salary_contribution') or None
                        participant_id_to_link = str(participant.participant_id)
                    else:
                        user = User.query.filter_by(email=participant_email).first()
                        
                        participant_args = {
                            'first_name': p_data['first_name'],
                            'last_name': p_data['last_name'],
                            'email': participant_email,
                            'phone': p_data['phone'],
                            'role': p_data['role'],
                            'affiliation': p_data['affiliation'],
                            'department': p_data['department'],
                            'orcid': p_data['orcid'],
                            'country': p_data['country'],
                            'funding_role': p_data['funding_role'],
                            'percent_effort': p_data.get('percent_effort') or None,
                            'salary_contribution': p_data.get('salary_contribution') or None
                        }
                        if user:
                            participant_args['participant_id'] = str(user.id)
                        
                        new_participant = StudyParticipant(**participant_args)
                        db.session.add(new_participant)
                        db.session.flush()
                        participant_id_to_link = str(new_participant.participant_id)

                    existing_link = StudyParticipantLink.query.filter_by(
                        study_id=str(new_study.study_id),
                        participant_id=participant_id_to_link
                    ).first()
                    
                    if not existing_link:
                        is_study_creator = (participant_email == current_user.email.lower())
                        
                        link = StudyParticipantLink(
                            study_id=str(new_study.study_id),
                            participant_id=participant_id_to_link,
                            active=True,
                            is_admin=is_study_creator
                        )
                        db.session.add(link)

                db.session.commit()
                flash('Study created successfully!', 'success')
                
                session.pop('new_study_type_id', None)
                session.pop('new_study_arms', None)
                session.pop('new_study_participants', None)
                session.pop('new_study_details', None)
                session.pop('new_study_settings', None)

                return redirect(url_for('main_app'))
            
            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred while saving the study: {e}', 'danger')
                return redirect(url_for('create_study_step3'))

    study_details = session.get('new_study_details', {})
    participants_data = session.get('new_study_participants', [])
    return render_template('create_study_step3.html', 
                           participants_data=participants_data, 
                           study_details=study_details)


@app.route('/edit_study/<study_id>', methods=['GET', 'POST'])
@login_required
def edit_study(study_id):
    study = Study.query.get_or_404(study_id)
    
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    is_admin = False # Default to not admin
    
    if participant:
        participant_id_str = str(participant.participant_id).lower()
        link = StudyParticipantLink.query.filter_by(
            study_id=study_id,
            participant_id=participant_id_str,
            is_admin=True
        ).first()
        if link:
            is_admin = True

    if request.method == 'POST':
        # Block non-admins from submitting updates
        if not is_admin:
            flash('You do not have permission to modify this study.', 'danger')
            return redirect(url_for('edit_study', study_id=study_id))

        study.name = request.form.get('name')
        study.description = request.form.get('description')
        study.status = request.form.get('status')
        study.funding_source = request.form.get('funding_source')
        study.currency = request.form.get('currency')
        start_date_str = request.form.get('start_date')
        study.start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
        end_date_str = request.form.get('end_date')
        study.end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None

        budget_str = request.form.get('budget_amount')
        new_budget_amount = None
        if budget_str:
            try:
                new_budget_amount = Decimal(budget_str)
            except InvalidOperation:
                flash('Invalid budget amount. Please enter a valid number.', 'danger')
                all_biomarkers = BiomarkerType.query.order_by(BiomarkerType.sample_type, BiomarkerType.biomarker_name).all()
                study_settings = StudySettings.query.get(study_id) or StudySettings(study_id=study_id, biological_enabled=True)
                selected_biomarker_ids = [str(b.biomarker_id) for b in study_settings.allowed_biomarkers]
                return render_template('edit_study.html', 
                                       study=study, 
                                       study_settings=study_settings,
                                       all_biomarkers=all_biomarkers,
                                       selected_biomarker_ids=json.dumps(selected_biomarker_ids),
                                       is_admin=is_admin)

        initial_budget_entry = FinancialLedger.query.filter_by(study_id=study_id, transaction_type='BUDGET').first()
        if initial_budget_entry:
            if new_budget_amount is not None and new_budget_amount != initial_budget_entry.amount:
                initial_budget_entry.amount = new_budget_amount
        elif new_budget_amount is not None and new_budget_amount > 0:
            new_entry = FinancialLedger(
                study_id=study_id,
                transaction_date=study.start_date or datetime.now(timezone.utc).date(),
                transaction_type='BUDGET',
                amount=new_budget_amount,
                description='Initial study budget'
            )
            db.session.add(new_entry)
        
        study.budget_amount = new_budget_amount
        db.session.commit()
        flash(f'Study "{study.name}" updated successfully!', 'success')
        return redirect(url_for('edit_study', study_id=study_id)) 

    all_biomarkers = BiomarkerType.query.order_by(BiomarkerType.sample_type, BiomarkerType.biomarker_name).all()
    study_settings = StudySettings.query.get(study_id)
    if not study_settings:
        study_settings = StudySettings(study_id=study_id, biological_enabled=True)

    selected_biomarker_ids = [str(b.biomarker_id) for b in study_settings.allowed_biomarkers]

    return render_template(
        'edit_study.html', 
        study=study,
        study_settings=study_settings,
        all_biomarkers=all_biomarkers,
        selected_biomarker_ids=json.dumps(selected_biomarker_ids),
        is_admin=is_admin
    )


@app.route('/study_settings/<study_id>/save', methods=['POST'])
@login_required
def save_study_settings(study_id):
    _require_study_membership(study_id)
    study = Study.query.get_or_404(study_id)
    settings = StudySettings.query.get(study_id)
    
    if not settings:
        settings = StudySettings(study_id=study_id)
        db.session.add(settings)

    try:
        settings.ai_enabled = request.form.get('ai_enabled') == 'true'
        settings.eeg_enabled = request.form.get('eeg_enabled') == 'true'
        settings.wearables_enabled = request.form.get('wearables_enabled') == 'true'
        settings.biological_enabled = request.form.get('biological_enabled') == 'true'
        settings.scales_enabled = request.form.get('scales_enabled') == 'true' # <-- ADDED
        
        if settings.ai_enabled:
            settings.ai_api_key = request.form.get('ai_api_key') or None
        else:
            settings.ai_api_key = None
            
        selected_ids = request.form.getlist('biomarker_types')
        if selected_ids:
            selected_biomarkers = BiomarkerType.query.filter(
                BiomarkerType.biomarker_id.in_(selected_ids)
            ).all()
            settings.allowed_biomarkers = selected_biomarkers
        else:
            settings.allowed_biomarkers = []
            
        db.session.commit()
        flash('Study settings updated successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while saving settings: {e}', 'danger')

    return redirect(url_for('edit_study', study_id=study_id))


@app.route('/manage_participants/<study_id>', methods=['GET', 'POST'])
@login_required
def manage_participants(study_id):
    """
    Route to add, edit, or remove participants from an existing study.
    """
    study = Study.query.get_or_404(study_id)

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'save_participant':
            participant_id = request.form.get('participant_id')
            is_admin_flag = request.form.get('is_admin') == 'true'
            
            participant_data = {
                'first_name': request.form.get('first_name', '').strip(),
                'last_name': request.form.get('last_name', '').strip(),
                'email': request.form.get('email', '').strip(),
                'phone': request.form.get('phone', '').strip(),
                'role': request.form.get('role', '').strip(),
                'affiliation': request.form.get('affiliation', '').strip(),
                'department': request.form.get('department', '').strip(),
                'orcid': request.form.get('orcid', '').strip(),
                'country': request.form.get('country', '').strip(),
                'funding_role': request.form.get('funding_role', '').strip(),
                'percent_effort': request.form.get('percent_effort') or None,
                'salary_contribution': request.form.get('salary_contribution') or None
            }

            if participant_id:
                participant = StudyParticipant.query.get(participant_id)
                if participant:
                    for key, value in participant_data.items():
                        setattr(participant, key, value)
                    link = StudyParticipantLink.query.filter_by(
                        study_id=study_id, 
                        participant_id=participant_id
                    ).first()
                    if link:
                        link.is_admin = is_admin_flag
                    db.session.commit()
                    flash('Participant updated successfully.', 'success')
            else:
                existing_participant = StudyParticipant.query.filter_by(email=participant_data['email']).first()
                if existing_participant:
                    link_exists = StudyParticipantLink.query.filter_by(study_id=study_id, participant_id=existing_participant.participant_id).first()
                    if not link_exists:
                        new_link = StudyParticipantLink(study_id=study_id, participant_id=str(existing_participant.participant_id).lower(), is_admin=is_admin_flag)
                        db.session.add(new_link)
                        db.session.commit()
                        flash('Existing participant added to the study.', 'success')
                    else:
                        flash('This participant is already in the study.', 'info')
                else:
                    new_participant = StudyParticipant(**participant_data)
                    db.session.add(new_participant)
                    db.session.flush()
                    new_link = StudyParticipantLink(study_id=study_id, participant_id=str(new_participant.participant_id).lower(), is_admin=is_admin_flag)
                    db.session.add(new_link)
                    db.session.commit()
                    flash('New participant added successfully.', 'success')
            return redirect(url_for('manage_participants', study_id=study_id))

        elif action == 'remove_participant':
            participant_id_to_remove = request.form.get('participant_id_to_remove')
            if participant_id_to_remove:
                link = StudyParticipantLink.query.filter_by(study_id=study_id, participant_id=participant_id_to_remove.lower()).first()
                if link:
                    db.session.delete(link)
                    db.session.commit()
                    flash('Participant removed from the study.', 'success')
            return redirect(url_for('manage_participants', study_id=study_id))

        elif action == 'save_changes':
            flash('Participant changes have been saved.', 'success')
            return redirect(url_for('edit_study', study_id=study_id))

    participant_links = StudyParticipantLink.query.filter_by(study_id=study_id).all()
    participant_ids = [link.participant_id for link in participant_links]
    admin_status_lookup = {link.participant_id.lower(): link.is_admin for link in participant_links if link.participant_id}
    participants_data = []
    
    if participant_ids:
        participants = StudyParticipant.query.filter(StudyParticipant.participant_id.in_(participant_ids)).all()
        
        for p in participants:
            participant_id_str = str(p.participant_id).lower()
            participants_data.append({
                'participant_id': participant_id_str,
                'first_name': p.first_name,
                'last_name': p.last_name,
                'email': p.email,
                'phone': p.phone,
                'role': p.role,
                'affiliation': p.affiliation,
                'department': p.department,
                'orcid': p.orcid,
                'country': p.country,
                'funding_role': p.funding_role,
                'percent_effort': str(p.percent_effort) if p.percent_effort is not None else '',
                'salary_contribution': str(p.salary_contribution) if p.salary_contribution is not None else '',
                'is_admin': bool(admin_status_lookup.get(participant_id_str, False))
            })

    return render_template('manage_participants.html', study=study, participants_data=participants_data)


@app.route('/revoke_study/<study_id>', methods=['POST'])
@login_required
def revoke_study(study_id):
    """
    Permanently deletes a study and all its associated data.
    """
    study = Study.query.get_or_404(study_id)
    study_name = study.name

    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    if not participant:
        flash('You do not have permission to delete this study.', 'danger')
        return redirect(url_for('main_app'))
    link = StudyParticipantLink.query.filter_by(
        study_id=study_id,
        participant_id=participant.participant_id,
        is_admin=True,
    ).first()
    if not link:
        flash('You do not have permission to delete this study.', 'danger')
        return redirect(url_for('main_app'))

    try:
        subjects = Subject.query.with_entities(Subject.subject_id).filter_by(study_id=study_id).all()
        subject_ids = [s[0] for s in subjects]

        if subject_ids:
            StudyRecordingBiomarker.query.filter(StudyRecordingBiomarker.study_id == study_id).delete(synchronize_session=False)
            StudyRecordingEEG.query.filter(StudyRecordingEEG.study_id == study_id).delete(synchronize_session=False)
            StudyRecordingWearable.query.filter(StudyRecordingWearable.study_id == study_id).delete(synchronize_session=False)
            StudyRecording.query.filter(StudyRecording.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectClinician.query.filter(SubjectClinician.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectConsent.query.filter(SubjectConsent.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectContact.query.filter(SubjectContact.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectDiagnosis.query.filter(SubjectDiagnosis.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectMedication.query.filter(SubjectMedication.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectDocument.query.filter(SubjectDocument.subject_id.in_(subject_ids)).delete(synchronize_session=False)

        Subject.query.filter_by(study_id=study_id).delete()
        StudyArm.query.filter_by(study_id=study_id).delete()
        StudyParticipantLink.query.filter_by(study_id=study_id).delete()
        StudySettings.query.filter_by(study_id=study_id).delete(synchronize_session=False)
        db.session.delete(study)
        db.session.commit()
        flash(f'The study "{study_name}" and all its associated records have been permanently revoked.', 'success')
    
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while revoking the study: {e}', 'danger')

    return redirect(url_for('main_app'))


@app.route('/edit_arms/<study_id>', methods=['GET', 'POST'])
@login_required
def edit_arms(study_id):
    """
    Route to manage arms for an existing study.
    """
    study = Study.query.get_or_404(study_id)

    if request.method == 'POST':
        _require_study_membership(study_id)
        action = request.form.get('action')
        arms_data = []
        i = 0
        while f'arm_name_{i}' in request.form:
            if action != f'remove_{i}':
                group_data = {
                    'arm_id': request.form.get(f'arm_id_{i}'),
                    'arm_name': request.form.get(f'arm_name_{i}', ''),
                    'description': request.form.get(f'description_{i}', '')
                }
                arms_data.append(group_data)
            i += 1

        if action == 'add_arm':
            arms_data.append({'arm_id': '', 'arm_name': '', 'description': ''})
            return render_template('edit_arms.html', study=study, arms_data=arms_data)
        
        elif action and action.startswith('remove_'):
            return render_template('edit_arms.html', study=study, arms_data=arms_data)
        
        elif action == 'save_changes':
            if not arms_data:
                flash('Please define at least one study arm.', 'warning')
                return render_template('edit_arms.html', study=study, arms_data=arms_data)
            
            for i, group in enumerate(arms_data):
                if not group.get('arm_name', '').strip() or not group.get('description', '').strip():
                    flash(f'Row {i+1} is incomplete. Please provide both an arm name and a description.', 'warning')
                    return render_template('edit_arms.html', study=study, arms_data=arms_data)
            
            existing_arms = StudyArm.query.filter_by(study_id=study_id).all()
            existing_arm_ids = {str(c.arm_id) for c in existing_arms}
            submitted_arm_ids = {g['arm_id'] for g in arms_data if g['arm_id']}

            try:
                arms_to_delete = [c for c in existing_arms if str(c.arm_id) not in submitted_arm_ids]
                for arm in arms_to_delete:
                    db.session.delete(arm)
                for group in arms_data:
                    arm_id = group.get('arm_id')
                    if arm_id and arm_id in existing_arm_ids:
                        arm_to_update = next((c for c in existing_arms if str(c.arm_id) == arm_id), None)
                        if arm_to_update:
                            arm_to_update.arm_name = group['arm_name']
                            arm_to_update.description = group['description']
                    else:
                        new_arm = StudyArm(
                            study_id=study_id,
                            arm_name=group['arm_name'],
                            description=group['description']
                        )
                        db.session.add(new_arm)
                
                db.session.commit()
                flash('Study arms updated successfully!', 'success')
                return redirect(url_for('edit_study', study_id=study_id))

            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred while updating arms: {e}', 'danger')
                return render_template('edit_arms.html', study=study, arms_data=arms_data)

    arms = StudyArm.query.filter_by(study_id=study_id).all()
    arms_data = [
        {'arm_id': str(c.arm_id), 'arm_name': c.arm_name, 'description': c.description}
        for c in arms
    ]
    if not arms_data:
        arms_data.append({'arm_id': '', 'arm_name': '', 'description': ''})

    return render_template('edit_arms.html', study=study, arms_data=arms_data)



@app.route('/manage_subjects/<study_id>', methods=['GET', 'POST'])
@login_required
def manage_subjects(study_id):
    """
    Route to manage subjects for an existing study.
    """
    study = Study.query.get_or_404(study_id)
    arms = StudyArm.query.filter_by(study_id=study_id).order_by(StudyArm.arm_name).all()

    if request.method == 'POST':
        _require_study_membership(study_id)
        action = request.form.get('action')
        if action == 'save_subject':
            subject_id = request.form.get('subject_id')
            external_code = request.form.get('external_subject_code', '').strip()
            if not external_code:
                flash('External Subject Code is a mandatory field.', 'danger')
                return redirect(url_for('manage_subjects', study_id=study_id))

            query = Subject.query.filter_by(study_id=study_id, external_subject_code=external_code)
            if subject_id:
                query = query.filter(Subject.subject_id != subject_id)
            
            if query.first():
                flash(f'The External Subject Code "{external_code}" is already in use for this study.', 'danger')
                return redirect(url_for('manage_subjects', study_id=study_id))

            try:
                subject_data = {
                    'external_subject_code': external_code,
                    'first_name': request.form.get('first_name') or None,
                    'last_name': request.form.get('last_name') or None,
                    'gender': request.form.get('gender') or None,
                    'ethnicity': request.form.get('ethnicity') or None,
                    'race': request.form.get('race') or None,
                    'handedness': request.form.get('handedness') or None,
                    'pregnancy_status': request.form.get('pregnancy_status') or None,
                    'city': request.form.get('city') or None,
                    'state_province': request.form.get('state_province') or None,
                    'country': request.form.get('country') or None,
                    'postal_code': request.form.get('postal_code') or None,
                    'education_level': request.form.get('education_level') or None,
                    'employment_status': request.form.get('employment_status') or None,
                    'marital_status': request.form.get('marital_status') or None,
                    'smoking_status': request.form.get('smoking_status') or None,
                    'alcohol_intake': request.form.get('alcohol_intake') or None,
                    'arm_id': request.form.get('arm_id') or None,
                    'physical_activity_level': request.form.get('physical_activity_level') or None,
                    'status': request.form.get('status') or None,
                    'site_identifier': request.form.get('site_identifier') or None,
                    'withdrawal_reason': request.form.get('withdrawal_reason') or None,
                    'screen_fail_reason': request.form.get('screen_fail_reason') or None,
                }

                for date_field in ['date_of_birth', 'consent_date', 'withdrawal_date', 'completion_date', 'screen_fail_date']:
                    date_str = request.form.get(date_field)
                    subject_data[date_field] = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None

                for numeric_field in ['height_cm', 'weight_kg']:
                    numeric_str = request.form.get(numeric_field)
                    subject_data[numeric_field] = Decimal(numeric_str) if numeric_str else None

                if subject_id:
                    subject = Subject.query.get(subject_id)
                    for key, value in subject_data.items():
                        setattr(subject, key, value)
                    flash('Subject updated successfully!', 'success')
                else:
                    subject_data['study_id'] = study_id
                    new_subject = Subject(**subject_data)
                    db.session.add(new_subject)
                    flash('Subject added successfully!', 'success')
                
                db.session.commit()

            except (InvalidOperation, ValueError) as e:
                db.session.rollback()
                flash(f'Invalid data provided. Please check numeric and date fields. Error: {e}', 'danger')
            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred: {e}', 'danger')

            return redirect(url_for('manage_subjects', study_id=study_id))
            
    subjects = db.session.query(
        Subject,
        StudyArm.arm_name
    ).outerjoin(
        StudyArm, Subject.arm_id == StudyArm.arm_id
    ).filter(
        Subject.study_id == study_id
    ).order_by(
        Subject.external_subject_code
    ).all()
    
    subjects_data = []
    for s, arm_name in subjects:
        subjects_data.append({
            'subject_id': str(s.subject_id),
            'arm_id': str(s.arm_id) if s.arm_id else '', 
            'arm_name': arm_name if arm_name else 'N/A', 
            'external_subject_code': s.external_subject_code,
            'first_name': s.first_name,
            'last_name': s.last_name,
            'gender': s.gender,
            'date_of_birth': s.date_of_birth.isoformat() if s.date_of_birth else None,
            'status': s.status,
            'ethnicity': s.ethnicity,
            'race': s.race,
            'handedness': s.handedness,
            'pregnancy_status': s.pregnancy_status,
            'city': s.city,
            'state_province': s.state_province,
            'country': s.country,
            'postal_code': s.postal_code,
            'education_level': s.education_level,
            'employment_status': s.employment_status,
            'marital_status': s.marital_status,
            'height_cm': str(s.height_cm) if s.height_cm is not None else '',
            'weight_kg': str(s.weight_kg) if s.weight_kg is not None else '',
            'smoking_status': s.smoking_status,
            'alcohol_intake': s.alcohol_intake,
            'physical_activity_level': s.physical_activity_level,
            'consent_date': s.consent_date.isoformat() if s.consent_date else None,
            'withdrawal_date': s.withdrawal_date.isoformat() if s.withdrawal_date else None,
            'completion_date': s.completion_date.isoformat() if s.completion_date else None,
            'site_identifier': s.site_identifier,
            'withdrawal_reason': s.withdrawal_reason,
            'screen_fail_date': s.screen_fail_date.isoformat() if s.screen_fail_date else None,
            'screen_fail_reason': s.screen_fail_reason,
        })
    return render_template('manage_subjects.html', study=study, subjects=subjects_data, arms=arms)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        topic = request.form.get('topic')
        message_body = request.form.get('message')

        if not name or not email or not message_body or not topic:
            flash('All fields are required.', 'danger')
            return redirect(url_for('contact'))

        try:
            subject = f"Contact Form: {topic} inquiry from {name}"
            msg = Message(subject,
                          sender=app.config['MAIL_DEFAULT_SENDER'],
                          recipients=['admin@novasyne.com'])
            msg.body = f"From: {name} <{email}>\nTopic: {topic}\n\n{message_body}"
            mail.send(msg)
            flash('Thank you for your message. We will get back to you shortly!', 'success')
        except Exception as e:
            app.logger.error(f"Mail sending failed: {e}")
            flash('Sorry, there was an error sending your message. Please try again later.', 'danger')

        return redirect(url_for('contact'))
        
    return render_template('contact.html')


@app.route('/study/<study_id>/documents', methods=['GET'])
@login_required
def get_study_documents(study_id):
    """API endpoint to get a JSON list of documents for a study."""
    study = Study.query.get_or_404(study_id)
    documents_list = [
        {
            'document_id': doc.document_id,
            'filename': doc.filename,
            'description': doc.description or 'N/A',
            'upload_date': doc.upload_date.strftime('%Y-%m-%d %H:%M'),
            'download_url': url_for('download_document', document_id=doc.document_id),
            'delete_url': url_for('delete_document', document_id=doc.document_id)
        } for doc in study.documents
    ]
    return jsonify(documents_list)


@app.route('/study/<study_id>/upload', methods=['POST'])
@login_required
def upload_document(study_id):
    """Handles file upload via AJAX and returns a JSON response."""
    _require_study_membership(study_id)
    if 'document' not in request.files or not request.files['document'].filename:
        return jsonify({'status': 'error', 'message': 'No file selected for upload.'}), 400

    file = request.files['document']
    if not _allowed_upload(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not permitted.'}), 400

    description = request.form.get('description', '').strip()

    if not description:
        return jsonify({'status': 'error', 'message': 'A description for the document is required.'}), 400
    
    try:
        data = file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            return jsonify({'status': 'error', 'message': 'File exceeds the 100 MB size limit.'}), 413
        new_doc = StudyDocument(
            study_id=study_id,
            filename=file.filename,
            description=description,
            data=data,
        )
        db.session.add(new_doc)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': f'Document "{file.filename}" was uploaded successfully.'
        })
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to upload document for study {study_id}: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred. Please try again.'}), 500


@app.route('/document/<document_id>/download', methods=['GET'])
@login_required
def download_document(document_id):
    """Serves a document from the database for download."""
    doc = StudyDocument.query.get_or_404(document_id)
    return send_file(
        BytesIO(doc.data),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=doc.filename
    )


@app.route('/document/<document_id>/delete', methods=['POST'])
@login_required
def delete_document(document_id):
    """Deletes a document from the database."""
    doc = StudyDocument.query.get_or_404(document_id)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Document deleted successfully.'})


@app.route('/subject/<subject_id>/documents', methods=['GET'])
@login_required
def get_subject_documents(subject_id):
    """API endpoint to get a JSON list of documents for a subject."""
    subject = Subject.query.get_or_404(subject_id)
    documents_list = [
        {
            'document_id': str(doc.document_id),
            'filename': doc.filename,
            'description': doc.description or 'N/A',
            'upload_date': doc.upload_date.strftime('%Y-%m-%d %H:%M'),
            'download_url': url_for('download_subject_document', document_id=str(doc.document_id)),
            'delete_url': url_for('delete_subject_document', document_id=str(doc.document_id))
        } for doc in subject.documents
    ]
    return jsonify(documents_list)


@app.route('/subject/<subject_id>/upload', methods=['POST'])
@login_required
def upload_subject_document(subject_id):
    """Handles file upload for a subject via AJAX and returns a JSON response."""
    subject = Subject.query.get_or_404(subject_id)
    _require_study_membership(str(subject.study_id))
    if 'document' not in request.files or not request.files['document'].filename:
        return jsonify({'status': 'error', 'message': 'No file selected for upload.'}), 400

    file = request.files['document']
    if not _allowed_upload(file.filename):
        return jsonify({'status': 'error', 'message': 'File type not permitted.'}), 400

    description = request.form.get('description', '').strip()

    if not description:
        return jsonify({'status': 'error', 'message': 'A description for the document is required.'}), 400

    try:
        data = file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            return jsonify({'status': 'error', 'message': 'File exceeds the 100 MB size limit.'}), 413
        new_doc = SubjectDocument(
            subject_id=subject_id,
            filename=file.filename,
            description=description,
            data=data,
        )
        db.session.add(new_doc)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to save subject document for {subject_id}: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while saving the document.'}), 500

    return jsonify({
        'status': 'success',
        'message': f'Document "{file.filename}" was uploaded successfully.'
    })


@app.route('/subject_document/<document_id>/download', methods=['GET'])
@login_required
def download_subject_document(document_id):
    """Serves a subject document from the database for download."""
    doc = SubjectDocument.query.get_or_404(document_id)
    return send_file(
        BytesIO(doc.data),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=doc.filename
    )


@app.route('/subject_document/<document_id>/delete', methods=['POST'])
@login_required
def delete_subject_document(document_id):
    """Deletes a subject document from the database."""
    doc = SubjectDocument.query.get_or_404(document_id)
    db.session.delete(doc)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Document deleted successfully.'})


@app.route('/subject/<subject_id>/consents', methods=['GET'])
@login_required
def get_subject_consents(subject_id):
    """API endpoint to get a JSON list of consents for a subject."""
    consents = SubjectConsent.query.filter_by(subject_id=subject_id).order_by(SubjectConsent.signed_at.desc()).all()
    consents_list = [
        {
            "consent_id": str(c.consent_id),
            "subject_id": str(c.subject_id),
            "consent_version": c.consent_version,
            "consent_type": c.consent_type,
            "signed_at": c.signed_at.strftime('%Y-%m-%d') if c.signed_at else None,
            "withdrawn_at": c.withdrawn_at.strftime('%Y-%m-%d') if c.withdrawn_at else None
        } for c in consents
    ]
    return jsonify({'data': consents_list})


@app.route('/subject/consent/save', methods=['POST'])
@login_required
def save_subject_consent():
    """Handles adding or editing a consent record via AJAX."""
    data = request.form
    subject_id = data.get('subject_id')
    consent_id = data.get('consent_id')

    if not subject_id:
        return jsonify({'status': 'error', 'message': 'Subject ID is required.'}), 400

    try:
        signed_at_val = datetime.strptime(data.get('signed_at'), '%Y-%m-%d') if data.get('signed_at') else None
        withdrawn_at_val = datetime.strptime(data.get('withdrawn_at'), '%Y-%m-%d') if data.get('withdrawn_at') else None

        if consent_id:
            consent = SubjectConsent.query.get(consent_id)
            if not consent:
                return jsonify({'status': 'error', 'message': 'Consent not found.'}), 404
            
            consent.consent_version = data.get('consent_version')
            consent.consent_type = data.get('consent_type')
            consent.signed_at = signed_at_val
            consent.withdrawn_at = withdrawn_at_val
            message = "Consent updated successfully."

        else:
            new_consent = SubjectConsent(
                subject_id=subject_id,
                consent_version=data.get('consent_version'),
                consent_type=data.get('consent_type'),
                signed_at=signed_at_val,
                withdrawn_at=withdrawn_at_val
            )
            db.session.add(new_consent)
            message = "Consent added successfully."
        
        db.session.commit()
        return jsonify({'status': 'success', 'message': message})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error saving consent: {e}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {e}'}), 500


@app.route('/subject/consent/delete/<consent_id>', methods=['POST'])
@login_required
def delete_subject_consent(consent_id):
    """Deletes a consent record."""
    consent = SubjectConsent.query.get_or_404(consent_id)
    try:
        db.session.delete(consent)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Consent deleted successfully.'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting consent: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while deleting.'}), 500


@app.route('/subject/<subject_id>/history', methods=['GET'])
@login_required
def get_subject_history(subject_id):
    """API endpoint to get all history data for a subject."""
    clinicians = SubjectClinician.query.filter_by(subject_id=subject_id).all()
    diagnoses = SubjectDiagnosis.query.filter_by(subject_id=subject_id).all()
    medications = SubjectMedication.query.filter_by(subject_id=subject_id).all()
    contacts = SubjectContact.query.filter_by(subject_id=subject_id).all()

    return jsonify({
        'clinicians': [{
            'clinician_id': c.clinician_id, 'first_name': c.first_name, 'last_name': c.last_name, 
            'specialty': c.specialty, 'organization': c.organization, 'city': c.city, 
            'country': c.country, 'email': c.email, 'phone': c.phone
        } for c in clinicians],
        'diagnoses': [{
            'id': d.id, 'diagnosis_code': d.diagnosis_code, 'diagnosis_description': d.diagnosis_description,
            'diagnosis_date': d.diagnosis_date.isoformat() if d.diagnosis_date else None, 
            'status': d.status, 'primary_diagnosis': d.primary_diagnosis
        } for d in diagnoses],
        'medications': [{
            'id': m.id, 'medication_name': m.medication_name, 'dose': m.dose, 'route': m.route,
            'start_date': m.start_date.isoformat() if m.start_date else None,
            'end_date': m.end_date.isoformat() if m.end_date else None,
            'indication': m.indication, 'currently_taking': m.currently_taking
        } for m in medications],
        'contacts': [{
            'contact_id': c.contact_id, 'contact_type': c.contact_type, 'contact_value': c.contact_value,
            'preferred': c.preferred, 'verified': c.verified
        } for c in contacts]
    })


@app.route('/subject/<subject_id>/history', methods=['POST'])
@login_required
def save_subject_history(subject_id):
    """API endpoint to save all history data for a subject."""
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid data provided.'}), 400

    try:
        SubjectClinician.query.filter_by(subject_id=subject_id).delete()
        SubjectDiagnosis.query.filter_by(subject_id=subject_id).delete()
        SubjectMedication.query.filter_by(subject_id=subject_id).delete()
        SubjectContact.query.filter_by(subject_id=subject_id).delete()

        for c in data.get('clinicians', []):
            if not (c.get('first_name') or c.get('last_name') or c.get('email')):
                continue
            db.session.add(SubjectClinician(
                subject_id=subject_id, first_name=c.get('first_name'), last_name=c.get('last_name'),
                specialty=c.get('specialty'), organization=c.get('organization'), city=c.get('city'),
                country=c.get('country'), email=c.get('email'), phone=c.get('phone')
            ))

        for d in data.get('diagnoses', []):
            if not (d.get('diagnosis_code') or d.get('diagnosis_description')):
                continue
            diag_date = datetime.strptime(d['diagnosis_date'], '%Y-%m-%d').date() if d.get('diagnosis_date') else None
            db.session.add(SubjectDiagnosis(
                subject_id=subject_id, diagnosis_code=d.get('diagnosis_code'),
                diagnosis_description=d.get('diagnosis_description'), diagnosis_date=diag_date,
                status=d.get('status'), primary_diagnosis=d.get('primary_diagnosis', False)
            ))

        for m in data.get('medications', []):
            if not m.get('medication_name'):
                continue
            start_date = datetime.strptime(m['start_date'], '%Y-%m-%d').date() if m.get('start_date') else None
            end_date = datetime.strptime(m['end_date'], '%Y-%m-%d').date() if m.get('end_date') else None
            db.session.add(SubjectMedication(
                subject_id=subject_id, medication_name=m.get('medication_name'), dose=m.get('dose'),
                route=m.get('route'), start_date=start_date, end_date=end_date,
                indication=m.get('indication'), currently_taking=m.get('currently_taking', False)
            ))

        for c in data.get('contacts', []):
            if not c.get('contact_value'):
                continue
            db.session.add(SubjectContact(
                subject_id=subject_id, contact_type=c.get('contact_type'), contact_value=c.get('contact_value'),
                preferred=c.get('preferred', False), verified=c.get('verified', False)
            ))

        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Subject history updated successfully.'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error saving subject history for {subject_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/study_recordings/<study_id>', methods=['GET'])
@login_required
def study_recordings(study_id):
    """
    Displays a list of recordings and data for the 'New Recording' modal.
    Filters modal dropdowns based on StudySettings.
    """
    study = Study.query.get_or_404(study_id)
    study_settings = StudySettings.query.get(study_id)
    if not study_settings:
        study_settings = StudySettings(study_id=study_id, biological_enabled=True, scales_enabled=False)
    
    subjects = Subject.query.filter_by(study_id=study_id).order_by('external_subject_code').all()

    all_allowed_biomarkers = study.settings.allowed_biomarkers if study.settings else []
    
    # Split them into two lists based on sample_type
    biological_biomarker_types = [
        b for b in all_allowed_biomarkers if b.sample_type.lower() != 'scale'
    ]
    scale_biomarker_types = [
        b for b in all_allowed_biomarkers if b.sample_type.lower() == 'scale'
    ]
    
    eeg_devices = EEG.query.order_by(EEG.manufacturer).all()
    wearable_devices = Wearable.query.order_by(Wearable.manufacturer).all()

    Subj = aliased(Subject)
    Biomarker = aliased(BiomarkerType)

    query = db.session.query(
        StudyRecording,
        Subj.external_subject_code,
        Subj.first_name,
        Subj.last_name,
        Biomarker.biomarker_name,
        StudyRecordingBiomarker.biomarker_value,
        SubjectSymptom.meddra_term.label('symptom_term'),
        SubjectSymptom.severity.label('symptom_severity'),
        SubjectAdverseEvent.meddra_term.label('ae_term'),
        SubjectAdverseEvent.severity_grade.label('ae_grade'),
        SubjectMedicationTaken.medication_name.label('med_name'),
        SubjectMedicationTaken.dose.label('med_dose'),
        case(
            (StudyRecordingEEG.data_uri != None, True),
            else_=False
        ).label('has_file_eeg'),
        case(
            (StudyRecordingWearable.data_uri != None, True),
            else_=False
        ).label('has_file_wearable'),
        case(
            (StudyRecordingImage.data_uri != None, True),
            else_=False
        ).label('has_file_image'),
        StudyRecordingImage.image_type.label('image_type')
    ).join(
        Subj, StudyRecording.subject_id == Subj.subject_id
    ).outerjoin(
        StudyRecordingBiomarker, StudyRecording.recording_id == StudyRecordingBiomarker.recording_id
    ).outerjoin(
        Biomarker, StudyRecordingBiomarker.biomarker_id == Biomarker.biomarker_id
    ).outerjoin(
        SubjectSymptom, StudyRecording.recording_id == SubjectSymptom.recording_id
    ).outerjoin(
        SubjectAdverseEvent, StudyRecording.recording_id == SubjectAdverseEvent.recording_id
    ).outerjoin(
        SubjectMedicationTaken, StudyRecording.recording_id == SubjectMedicationTaken.recording_id
    ).outerjoin(
        StudyRecordingEEG, StudyRecording.recording_id == StudyRecordingEEG.recording_id
    ).outerjoin(
        StudyRecordingWearable, StudyRecording.recording_id == StudyRecordingWearable.recording_id
    ).outerjoin(
        StudyRecordingImage, StudyRecording.recording_id == StudyRecordingImage.recording_id
    ).filter(
        StudyRecording.study_id == study_id
    ).order_by(
        desc(StudyRecording.recording_datetime)
    )

    recordings_list = []
    for rec, code, fname, lname, b_name, b_val, s_term, s_sev, ae_term, ae_grade, m_name, m_dose, has_eeg, has_wearable, has_image, img_type in query.all():
        recordings_list.append({
            'recording_id': rec.recording_id,
            'subject_code': code,
            'first_name': fname,
            'last_name': lname,
            'recording_type': rec.recording_type,
            'recording_datetime': rec.recording_datetime.isoformat(),
            'biomarker_name': b_name,
            'biomarker_value': b_val,
            'symptom_term': s_term,
            'symptom_severity': s_sev,
            'ae_term': ae_term,
            'ae_grade': ae_grade,
            'med_name': m_name,
            'med_dose': m_dose,
            'image_type': img_type,
            'has_file': has_eeg or has_wearable or has_image
        })

    return render_template('study_recordings.html',
                           study=study,
                           study_settings=study_settings,
                           subjects=subjects,
                           recordings=recordings_list,
                           biological_biomarker_types=biological_biomarker_types,
                           scale_biomarker_types=scale_biomarker_types,
                           eeg_devices=eeg_devices,
                           wearable_devices=wearable_devices)


@app.route('/recording/<recording_id>/delete', methods=['POST'])
@login_required
def delete_recording(recording_id):
    """ Deletes a recording from the database and its associated file from Azure Blob Storage if applicable. """
    recording = StudyRecording.query.get_or_404(recording_id)
    study_id_for_redirect = recording.study_id
    data_uri = None

    try:
        if recording.recording_type == 'EEG':
            eeg_data = StudyRecordingEEG.query.get(recording_id)
            if eeg_data: data_uri = eeg_data.data_uri
        elif recording.recording_type == 'Wearable':
            wearable_data = StudyRecordingWearable.query.get(recording_id)
            if wearable_data: data_uri = wearable_data.data_uri
        elif recording.recording_type == 'Imaging':
            image_data = StudyRecordingImage.query.get(recording_id)
            if image_data: data_uri = image_data.data_uri

        if data_uri:
            connect_str = os.environ.get('AZURE_BLOB')
            if connect_str:
                try:
                    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                    container_name = data_uri.split('/')[3]
                    blob_name = '/'.join(data_uri.split('/')[4:])
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                    blob_client.delete_blob()
                except ResourceNotFoundError:
                    flash('File was not found in cloud storage, but deleting database record.', 'warning')
                except Exception as e:
                    app.logger.error(f"Error deleting blob {data_uri}: {e}")
                    flash('Could not delete file from cloud storage. Please remove it manually.', 'danger')
            else:
                flash('Azure connection string not configured. Cannot delete file.', 'warning')

        db.session.delete(recording)
        db.session.commit()
        flash('Recording has been successfully deleted.', 'success')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting recording {recording_id}: {e}")
        flash(f'An error occurred while deleting the recording: {e}', 'danger')

    return redirect(url_for('study_recordings', study_id=study_id_for_redirect))


@app.route('/recording/<recording_id>/download', methods=['GET'])
@login_required
def download_recording(recording_id):
    """ Generates a secure SAS URL and redirects the user to download the file from Azure. """
    recording = StudyRecording.query.get_or_404(recording_id)
    data_uri = None

    if recording.recording_type == 'EEG':
        eeg_data = StudyRecordingEEG.query.get(recording_id)
        if eeg_data: data_uri = eeg_data.data_uri
    elif recording.recording_type == 'Wearable':
        wearable_data = StudyRecordingWearable.query.get(recording_id)
        if wearable_data: data_uri = wearable_data.data_uri
    elif recording.recording_type == 'Imaging':
        img_data = StudyRecordingImage.query.get(recording_id)
        if img_data: data_uri = img_data.data_uri

    if not data_uri:
        flash('No downloadable file associated with this recording.', 'danger')
        return redirect(url_for('study_recordings', study_id=recording.study_id))
    
    connect_str = os.environ.get('AZURE_BLOB')
    if not connect_str:
        flash('Azure connection string not configured. Cannot generate download link.', 'danger')
        return redirect(url_for('study_recordings', study_id=recording.study_id))

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_name = "recordings"
        blob_name = '/'.join(data_uri.split('/')[4:])

        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=1)
        )

        sas_url = f"{data_uri}?{sas_token}"
        return redirect(sas_url)

    except Exception as e:
        app.logger.error(f"Error generating SAS URL for {data_uri}: {e}")
        flash('Could not generate a secure download link.', 'danger')
        return redirect(url_for('study_recordings', study_id=recording.study_id))


@app.route('/study/<study_id>/add_recording', methods=['POST'])
@login_required
def add_recording(study_id):
    """
    Handles the complex "Add New Recording" form.
    This one route handles all recording types including Imaging.
    """
    _require_study_membership(study_id)
    Study.query.get_or_404(study_id)
    form = request.form
    
    subject_id = form.get('subject_id')
    recording_datetime_str = form.get('recording_datetime')
    recording_type = form.get('recording_type')

    if not all([subject_id, recording_datetime_str, recording_type]):
        flash('Missing required fields: Subject, Date/Time, and Type.', 'danger')
        return redirect(url_for('study_recordings', study_id=study_id))

    try:
        recording_datetime = datetime.fromisoformat(recording_datetime_str)

        new_rec = StudyRecording(
            study_id=study_id,
            subject_id=subject_id,
            recording_datetime=recording_datetime,
            recording_type=recording_type
        )
        db.session.add(new_rec)

        if recording_type in ('Biomarker', 'Scale'):
            biomarker_rec = StudyRecordingBiomarker(
                recording=new_rec,
                study_id=study_id,
                subject_id=subject_id,
                biomarker_id=form.get('biomarker_id'),
                biomarker_value=form.get('biomarker_value')
            )
            db.session.add(biomarker_rec)

        elif recording_type == 'Symptom':
            symptom_rec = SubjectSymptom(
                recording=new_rec,
                study_id=study_id,
                subject_id=subject_id,
                symptom_verbatim=form.get('symptom_verbatim'),
                meddra_code=form.get('symptom_meddra_code'),
                meddra_term=form.get('symptom_meddra_term'),
                severity=form.get('symptom_severity')
            )
            db.session.add(symptom_rec)
        
        elif recording_type == 'Adverse Event':
            is_serious = form.get('is_serious_ae') == 'true'
            
            ae_rec = SubjectAdverseEvent(
                recording=new_rec,
                study_id=study_id,
                subject_id=subject_id,
                ae_verbatim=form.get('ae_verbatim'),
                meddra_code=form.get('ae_meddra_code'),
                meddra_term=form.get('ae_meddra_term'),
                is_serious_ae=is_serious,
                severity_grade=form.get('severity_grade'),
                causality=form.get('causality'),
                outcome=form.get('outcome')
            )
            db.session.add(ae_rec)

        elif recording_type == 'Medication':
            is_concomitant = form.get('is_concomitant') == 'true'

            med_rec = SubjectMedicationTaken(
                recording=new_rec,
                study_id=study_id,
                subject_id=subject_id,
                medication_name=form.get('medication_name'),
                dose=form.get('medication_dose'),
                route=form.get('medication_route'),
                indication=form.get('medication_indication'),
                is_concomitant=is_concomitant
            )
            db.session.add(med_rec)

        elif recording_type in ['EEG', 'Wearable', 'Imaging']:
            file = request.files.get('recording_file')
            if not file:
                raise ValueError(f"No file uploaded for {recording_type} recording.")

            connect_str = os.environ.get('AZURE_BLOB')
            if not connect_str:
                raise RuntimeError("Azure connection string not configured.")

            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_name = "recordings"
            try:
                blob_service_client.create_container(container_name)
            except Exception:
                pass # Container already exists

            # For Imaging, we preserve the filename extension so detection works later.
            # For EEG/Wearable, we default to .zip (as per existing logic).
            if recording_type == 'Imaging':
                blob_name = f"{study_id}/{subject_id}/{str(uuid.uuid4())}_{file.filename}"
            else:
                blob_name = f"{study_id}/{subject_id}/{str(uuid.uuid4())}.zip"

            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            blob_client.upload_blob(file.stream, overwrite=True)
            data_uri = blob_client.url # This is the full URL
            
            if recording_type == 'EEG':
                eeg_rec = StudyRecordingEEG(
                    recording=new_rec,
                    study_id=study_id,
                    subject_id=subject_id,
                    data_uri=data_uri,
                    eeg_id=form.get('eeg_id') or None
                )
                db.session.add(eeg_rec)
            
            elif recording_type == 'Wearable':
                wearable_rec = StudyRecordingWearable(
                    recording=new_rec,
                    study_id=study_id,
                    subject_id=subject_id,
                    data_uri=data_uri,
                    wearable_id=form.get('wearable_id') or None
                )
                db.session.add(wearable_rec)

            elif recording_type == 'Imaging':
                image_rec = StudyRecordingImage(
                    recording=new_rec,
                    study_id=study_id,
                    subject_id=subject_id,
                    data_uri=data_uri,
                    image_type=form.get('image_type') or 'Unknown'
                )
                db.session.add(image_rec)

        db.session.commit()
        flash(f'{recording_type} recording added successfully.', 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'Error adding recording: {e}', 'danger')

    return redirect(url_for('study_recordings', study_id=study_id))


@app.route('/export_study/<study_id>', methods=['GET'])
@login_required
def export_study(study_id):
    """
    Exports ALL study data including database tables and blob storage files as a single ZIP file.
    
    Exported structure:
        study_metadata.csv          - Study details and type
        study_settings.csv          - Study configuration flags
        study_arms.csv              - Study arm definitions
        subjects.csv                - Full subject demographics
        subject_clinicians.csv      - Clinician assignments per subject
        subject_consents.csv        - Consent records
        subject_contacts.csv        - Subject contact information
        subject_diagnoses.csv       - Subject diagnoses
        subject_medications.csv     - Subject medication history
        recordings.csv              - Base recording sessions
        recordings_biomarkers.csv   - Biomarker values with type info
        recordings_eeg.csv          - EEG recording metadata
        recordings_wearable.csv     - Wearable recording metadata
        recordings_imaging.csv      - Imaging recording metadata
        symptoms.csv                - Reported symptoms (MedDRA coded)
        adverse_events.csv          - Adverse events (MedDRA coded)
        medications_taken.csv       - Concomitant medications taken per recording
        financial_ledger.csv        - Financial transactions and budget
        audit_log.csv               - Full audit trail
        DOCUMENTS/                  - Study-level document files
        SUBJECT_DOCUMENTS/          - Subject-level document files
        EEG/                        - EEG blob storage files
        WEARABLE/                   - Wearable blob storage files
        IMAGING/                    - Imaging blob storage files
        KNOWLEDGE/                  - Uploaded knowledge PDF files
    """
    study = Study.query.get_or_404(study_id)

    memory_file = BytesIO()

    connect_str = os.environ.get('AZURE_BLOB')
    container_client = None
    if connect_str:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client("recordings")
        except Exception as e:
            app.logger.error(f"Failed to connect to Azure Blob Storage: {e}")

    def download_blob_to_zip(zf, uri, folder_name):
        """Helper to download a blob from Azure and write it into the ZIP under the given folder."""
        try:
            blob_name = '/'.join(uri.split('/')[4:])
            file_name = os.path.basename(blob_name)
            blob_client = container_client.get_blob_client(blob_name)
            downloader = blob_client.download_blob()
            zf.writestr(f'{folder_name}/{file_name}', downloader.readall())
        except Exception as e:
            app.logger.error(f"Failed to download blob {uri}: {e}")
            zf.writestr(f'{folder_name}/ERROR_downloading_{os.path.basename(uri)}.txt', str(e))

    def write_csv_to_zip(zf, filename, query_or_df):
        """Helper to execute a query (or accept a DataFrame) and write it as CSV into the ZIP."""
        if isinstance(query_or_df, pd.DataFrame):
            df = query_or_df
        else:
            df = pd.read_sql(query_or_df.statement, db.engine)
        if not df.empty:
            zf.writestr(filename, df.to_csv(index=False, encoding='utf-8'))

    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:

        # =====================================================================
        # 1. STUDY METADATA
        # =====================================================================
        study_query = db.session.query(
            Study.study_id, Study.name, Study.description,
            Study.start_date, Study.end_date, Study.status,
            Study.funding_source, Study.budget_amount, Study.currency,
            Study.principal_investigator_id,
            Study.created_at, Study.updated_at,
            StudyType.category.label('study_type_category'),
            StudyType.study_type.label('study_type')
        ).outerjoin(
            StudyType, Study.study_type_id == StudyType.study_type_id
        ).filter(Study.study_id == study_id)

        write_csv_to_zip(zf, 'study_metadata.csv', study_query)

        # =====================================================================
        # 2. STUDY SETTINGS
        # =====================================================================
        study_settings_query = db.session.query(
            StudySettings.study_id,
            StudySettings.ai_enabled,
            StudySettings.eeg_enabled,
            StudySettings.wearables_enabled,
            StudySettings.biological_enabled,
            StudySettings.scales_enabled
        ).filter(StudySettings.study_id == study_id)
        # Note: ai_api_key is intentionally excluded for security

        write_csv_to_zip(zf, 'study_settings.csv', study_settings_query)

        # Allowed biomarkers for this study
        allowed_biomarkers_query = db.session.query(
            study_settings_biomarker_types.c.study_id,
            study_settings_biomarker_types.c.biomarker_id,
            BiomarkerType.biomarker_name,
            BiomarkerType.sample_type,
            BiomarkerType.category_notes
        ).join(
            BiomarkerType,
            study_settings_biomarker_types.c.biomarker_id == BiomarkerType.biomarker_id
        ).filter(
            study_settings_biomarker_types.c.study_id == study_id
        )

        write_csv_to_zip(zf, 'study_settings_allowed_biomarkers.csv', allowed_biomarkers_query)

        # =====================================================================
        # 3. STUDY ARMS
        # =====================================================================
        arms_query = db.session.query(
            StudyArm.arm_id, StudyArm.study_id,
            StudyArm.arm_name, StudyArm.description
        ).filter(StudyArm.study_id == study_id).order_by(StudyArm.arm_name)

        write_csv_to_zip(zf, 'study_arms.csv', arms_query)

        # =====================================================================
        # 4. SUBJECTS (full demographics)
        # =====================================================================
        subjects_query = db.session.query(
            Subject.subject_id, Subject.external_subject_code,
            Subject.study_id, Subject.arm_id,
            StudyArm.arm_name,
            Subject.status, Subject.enrollment_date, Subject.completion_date,
            Subject.screen_fail_date, Subject.screen_fail_reason,
            Subject.withdrawal_reason, Subject.site_identifier,
            Subject.first_name, Subject.last_name,
            Subject.gender, Subject.date_of_birth,
            Subject.ethnicity, Subject.race, Subject.handedness,
            Subject.pregnancy_status,
            Subject.city, Subject.state_province, Subject.country, Subject.postal_code,
            Subject.education_level, Subject.employment_status, Subject.marital_status,
            Subject.height_cm, Subject.weight_kg,
            Subject.smoking_status, Subject.alcohol_intake, Subject.physical_activity_level,
            Subject.consent_date, Subject.withdrawal_date,
            Subject.created_at, Subject.updated_at
        ).outerjoin(
            StudyArm, Subject.arm_id == StudyArm.arm_id
        ).filter(Subject.study_id == study_id).order_by(Subject.external_subject_code)

        write_csv_to_zip(zf, 'subjects.csv', subjects_query)

        # =====================================================================
        # 5. SUBJECT CLINICIANS
        # =====================================================================
        # Get all subject_ids for this study first
        study_subject_ids = db.session.query(Subject.subject_id).filter(
            Subject.study_id == study_id
        ).subquery()

        clinicians_query = db.session.query(
            SubjectClinician.clinician_id, SubjectClinician.subject_id,
            Subject.external_subject_code,
            SubjectClinician.first_name, SubjectClinician.last_name,
            SubjectClinician.specialty, SubjectClinician.organization,
            SubjectClinician.city, SubjectClinician.country,
            SubjectClinician.email, SubjectClinician.phone
        ).join(
            Subject, SubjectClinician.subject_id == Subject.subject_id
        ).filter(SubjectClinician.subject_id.in_(study_subject_ids))

        write_csv_to_zip(zf, 'subject_clinicians.csv', clinicians_query)

        # =====================================================================
        # 6. SUBJECT CONSENTS
        # =====================================================================
        consents_query = db.session.query(
            SubjectConsent.consent_id, SubjectConsent.subject_id,
            Subject.external_subject_code,
            SubjectConsent.consent_version, SubjectConsent.consent_type,
            SubjectConsent.signed_at, SubjectConsent.withdrawn_at
        ).join(
            Subject, SubjectConsent.subject_id == Subject.subject_id
        ).filter(SubjectConsent.subject_id.in_(study_subject_ids))

        write_csv_to_zip(zf, 'subject_consents.csv', consents_query)

        # =====================================================================
        # 7. SUBJECT CONTACTS
        # =====================================================================
        contacts_query = db.session.query(
            SubjectContact.contact_id, SubjectContact.subject_id,
            Subject.external_subject_code,
            SubjectContact.contact_type, SubjectContact.contact_value,
            SubjectContact.preferred, SubjectContact.verified
        ).join(
            Subject, SubjectContact.subject_id == Subject.subject_id
        ).filter(SubjectContact.subject_id.in_(study_subject_ids))

        write_csv_to_zip(zf, 'subject_contacts.csv', contacts_query)

        # =====================================================================
        # 8. SUBJECT DIAGNOSES
        # =====================================================================
        diagnoses_query = db.session.query(
            SubjectDiagnosis.id, SubjectDiagnosis.subject_id,
            Subject.external_subject_code,
            SubjectDiagnosis.diagnosis_code, SubjectDiagnosis.diagnosis_description,
            SubjectDiagnosis.diagnosis_date, SubjectDiagnosis.status,
            SubjectDiagnosis.primary_diagnosis
        ).join(
            Subject, SubjectDiagnosis.subject_id == Subject.subject_id
        ).filter(SubjectDiagnosis.subject_id.in_(study_subject_ids))

        write_csv_to_zip(zf, 'subject_diagnoses.csv', diagnoses_query)

        # =====================================================================
        # 9. SUBJECT MEDICATIONS (medical history)
        # =====================================================================
        medications_query = db.session.query(
            SubjectMedication.id, SubjectMedication.subject_id,
            Subject.external_subject_code,
            SubjectMedication.medication_name, SubjectMedication.dose,
            SubjectMedication.route, SubjectMedication.start_date,
            SubjectMedication.end_date, SubjectMedication.indication,
            SubjectMedication.currently_taking
        ).join(
            Subject, SubjectMedication.subject_id == Subject.subject_id
        ).filter(SubjectMedication.subject_id.in_(study_subject_ids))

        write_csv_to_zip(zf, 'subject_medications.csv', medications_query)

        # =====================================================================
        # 10. BASE RECORDINGS
        # =====================================================================
        recordings_query = db.session.query(
            StudyRecording.recording_id, StudyRecording.study_id,
            StudyRecording.subject_id,
            Subject.external_subject_code,
            StudyArm.arm_name,
            StudyRecording.recording_datetime,
            StudyRecording.recording_type
        ).join(
            Subject, StudyRecording.subject_id == Subject.subject_id
        ).outerjoin(
            StudyArm, Subject.arm_id == StudyArm.arm_id
        ).filter(
            StudyRecording.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        write_csv_to_zip(zf, 'recordings.csv', recordings_query)

        # =====================================================================
        # 11. RECORDING BIOMARKERS
        # =====================================================================
        biomarkers_query = db.session.query(
            StudyRecordingBiomarker.recording_id,
            StudyRecordingBiomarker.study_id,
            StudyRecordingBiomarker.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            BiomarkerType.biomarker_name,
            BiomarkerType.sample_type,
            StudyRecordingBiomarker.biomarker_value
        ).join(
            Subject, StudyRecordingBiomarker.subject_id == Subject.subject_id
        ).join(
            StudyRecording, StudyRecordingBiomarker.recording_id == StudyRecording.recording_id
        ).join(
            BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id
        ).filter(
            StudyRecordingBiomarker.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        write_csv_to_zip(zf, 'recordings_biomarkers.csv', biomarkers_query)

        # =====================================================================
        # 12. RECORDING EEG
        # =====================================================================
        eeg_query = db.session.query(
            StudyRecordingEEG.recording_id,
            StudyRecordingEEG.study_id,
            StudyRecordingEEG.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            EEG.manufacturer.label('eeg_manufacturer'),
            EEG.device_type.label('eeg_device_type'),
            StudyRecordingEEG.data_uri.label('eeg_data_uri')
        ).join(
            Subject, StudyRecordingEEG.subject_id == Subject.subject_id
        ).join(
            StudyRecording, StudyRecordingEEG.recording_id == StudyRecording.recording_id
        ).outerjoin(
            EEG, StudyRecordingEEG.eeg_id == EEG.eeg_id
        ).filter(
            StudyRecordingEEG.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        df_eeg = pd.read_sql(eeg_query.statement, db.engine)
        if not df_eeg.empty:
            zf.writestr('recordings_eeg.csv', df_eeg.to_csv(index=False, encoding='utf-8'))
            if container_client:
                for uri in df_eeg['eeg_data_uri'].dropna().unique():
                    download_blob_to_zip(zf, uri, 'EEG')

        # =====================================================================
        # 13. RECORDING WEARABLE
        # =====================================================================
        wearable_query = db.session.query(
            StudyRecordingWearable.recording_id,
            StudyRecordingWearable.study_id,
            StudyRecordingWearable.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            Wearable.manufacturer.label('wearable_manufacturer'),
            Wearable.device_name.label('wearable_device_name'),
            Wearable.wearable_type.label('wearable_type'),
            Wearable.wearable_location.label('wearable_location'),
            StudyRecordingWearable.data_uri.label('wearable_data_uri')
        ).join(
            Subject, StudyRecordingWearable.subject_id == Subject.subject_id
        ).join(
            StudyRecording, StudyRecordingWearable.recording_id == StudyRecording.recording_id
        ).outerjoin(
            Wearable, StudyRecordingWearable.wearable_id == Wearable.wearable_id
        ).filter(
            StudyRecordingWearable.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        df_wearable = pd.read_sql(wearable_query.statement, db.engine)
        if not df_wearable.empty:
            zf.writestr('recordings_wearable.csv', df_wearable.to_csv(index=False, encoding='utf-8'))
            if container_client:
                for uri in df_wearable['wearable_data_uri'].dropna().unique():
                    download_blob_to_zip(zf, uri, 'WEARABLE')

        # =====================================================================
        # 14. RECORDING IMAGING
        # =====================================================================
        imaging_query = db.session.query(
            StudyRecordingImage.recording_id,
            StudyRecordingImage.study_id,
            StudyRecordingImage.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            StudyRecordingImage.image_type,
            StudyRecordingImage.data_uri.label('image_data_uri')
        ).join(
            Subject, StudyRecordingImage.subject_id == Subject.subject_id
        ).join(
            StudyRecording, StudyRecordingImage.recording_id == StudyRecording.recording_id
        ).filter(
            StudyRecordingImage.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        df_imaging = pd.read_sql(imaging_query.statement, db.engine)
        if not df_imaging.empty:
            zf.writestr('recordings_imaging.csv', df_imaging.to_csv(index=False, encoding='utf-8'))
            if container_client:
                for uri in df_imaging['image_data_uri'].dropna().unique():
                    download_blob_to_zip(zf, uri, 'IMAGING')

        # =====================================================================
        # 15. SYMPTOMS
        # =====================================================================
        symptoms_query = db.session.query(
            SubjectSymptom.recording_id,
            SubjectSymptom.study_id,
            SubjectSymptom.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            SubjectSymptom.symptom_verbatim,
            SubjectSymptom.meddra_code,
            SubjectSymptom.meddra_term,
            SubjectSymptom.severity,
            MedDRA.soc_name.label('meddra_soc'),
            MedDRA.hlt_name.label('meddra_hlt'),
            MedDRA.pt_name.label('meddra_pt')
        ).join(
            Subject, SubjectSymptom.subject_id == Subject.subject_id
        ).join(
            StudyRecording, SubjectSymptom.recording_id == StudyRecording.recording_id
        ).outerjoin(
            MedDRA, SubjectSymptom.meddra_code == MedDRA.meddra_code
        ).filter(
            SubjectSymptom.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        write_csv_to_zip(zf, 'symptoms.csv', symptoms_query)

        # =====================================================================
        # 16. ADVERSE EVENTS
        # =====================================================================
        ae_query = db.session.query(
            SubjectAdverseEvent.recording_id,
            SubjectAdverseEvent.study_id,
            SubjectAdverseEvent.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            SubjectAdverseEvent.ae_verbatim,
            SubjectAdverseEvent.meddra_code,
            SubjectAdverseEvent.meddra_term,
            SubjectAdverseEvent.is_serious_ae,
            SubjectAdverseEvent.severity_grade,
            SubjectAdverseEvent.causality,
            SubjectAdverseEvent.outcome,
            MedDRA.soc_name.label('meddra_soc'),
            MedDRA.hlt_name.label('meddra_hlt'),
            MedDRA.pt_name.label('meddra_pt')
        ).join(
            Subject, SubjectAdverseEvent.subject_id == Subject.subject_id
        ).join(
            StudyRecording, SubjectAdverseEvent.recording_id == StudyRecording.recording_id
        ).outerjoin(
            MedDRA, SubjectAdverseEvent.meddra_code == MedDRA.meddra_code
        ).filter(
            SubjectAdverseEvent.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        write_csv_to_zip(zf, 'adverse_events.csv', ae_query)

        # =====================================================================
        # 17. MEDICATIONS TAKEN (concomitant meds per recording)
        # =====================================================================
        meds_taken_query = db.session.query(
            SubjectMedicationTaken.recording_id,
            SubjectMedicationTaken.study_id,
            SubjectMedicationTaken.subject_id,
            Subject.external_subject_code,
            StudyRecording.recording_datetime,
            SubjectMedicationTaken.medication_name,
            SubjectMedicationTaken.dose,
            SubjectMedicationTaken.route,
            SubjectMedicationTaken.indication,
            SubjectMedicationTaken.is_concomitant
        ).join(
            Subject, SubjectMedicationTaken.subject_id == Subject.subject_id
        ).join(
            StudyRecording, SubjectMedicationTaken.recording_id == StudyRecording.recording_id
        ).filter(
            SubjectMedicationTaken.study_id == study_id
        ).order_by(StudyRecording.recording_datetime)

        write_csv_to_zip(zf, 'medications_taken.csv', meds_taken_query)

        # =====================================================================
        # 18. FINANCIAL LEDGER
        # =====================================================================
        financial_query = db.session.query(
            FinancialLedger.transaction_id,
            FinancialLedger.study_id,
            FinancialLedger.transaction_date,
            FinancialLedger.transaction_type,
            FinancialLedger.amount,
            FinancialLedger.description,
            ExpenseCategory.category_name.label('expense_category'),
            FinancialLedger.created_at
        ).outerjoin(
            ExpenseCategory, FinancialLedger.category_id == ExpenseCategory.category_id
        ).filter(
            FinancialLedger.study_id == study_id
        ).order_by(FinancialLedger.transaction_date)

        write_csv_to_zip(zf, 'financial_ledger.csv', financial_query)

        # =====================================================================
        # 19. AUDIT LOG
        # =====================================================================
        audit_query = db.session.query(
            AuditLog.audit_log_id,
            AuditLog.change_datetime,
            AuditLog.user_email,
            AuditLog.study_id,
            AuditLog.subject_id,
            Subject.external_subject_code,
            AuditLog.record_id,
            AuditLog.changed_table,
            AuditLog.change_type,
            AuditLog.operation_type,
            AuditLog.old_value,
            AuditLog.new_value
        ).outerjoin(
            Subject, AuditLog.subject_id == Subject.subject_id
        ).filter(
            AuditLog.study_id == study_id
        ).order_by(AuditLog.change_datetime)

        write_csv_to_zip(zf, 'audit_log.csv', audit_query)

        # =====================================================================
        # 20. STUDY DOCUMENTS (binary files)
        # =====================================================================
        study_docs = StudyDocument.query.filter_by(study_id=study_id).all()
        for doc in study_docs:
            try:
                zf.writestr(f'DOCUMENTS/{doc.filename}', doc.data)
            except Exception as e:
                app.logger.error(f"Failed to write study document {doc.filename}: {e}")
                zf.writestr(f'DOCUMENTS/ERROR_{doc.filename}.txt', str(e))

        # =====================================================================
        # 21. SUBJECT DOCUMENTS (binary files)
        # =====================================================================
        subject_docs = db.session.query(
            SubjectDocument, Subject.external_subject_code
        ).join(
            Subject, SubjectDocument.subject_id == Subject.subject_id
        ).filter(
            SubjectDocument.subject_id.in_(study_subject_ids)
        ).all()

        for doc, subject_code in subject_docs:
            try:
                safe_code = subject_code or str(doc.subject_id)
                zf.writestr(f'SUBJECT_DOCUMENTS/{safe_code}/{doc.filename}', doc.data)
            except Exception as e:
                app.logger.error(f"Failed to write subject document {doc.filename}: {e}")
                zf.writestr(f'SUBJECT_DOCUMENTS/ERROR_{doc.filename}.txt', str(e))

        # =====================================================================
        # 22. KNOWLEDGE FILES (PDF files)
        # =====================================================================
        knowledge_files = StudyKnowledge.query.filter_by(study_id=study_id).all()
        for kf in knowledge_files:
            try:
                zf.writestr(f'KNOWLEDGE/{kf.filename}', kf.data)
            except Exception as e:
                app.logger.error(f"Failed to write knowledge file {kf.filename}: {e}")
                zf.writestr(f'KNOWLEDGE/ERROR_{kf.filename}.txt', str(e))

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{study.name.replace(' ', '_')}_full_export.zip"
    )


@app.route('/analytics/<study_id>', methods=['GET'])
@login_required
def analytics(study_id):
    """Renders the main analytics page and populates its dropdowns."""
    study = Study.query.get_or_404(study_id)

    study_settings = StudySettings.query.get(study_id)
    ai_available = False
    if study_settings and study_settings.ai_enabled and study_settings.ai_api_key:
        ai_available = True
    
    arms = StudyArm.query.filter_by(study_id=study_id).order_by(StudyArm.arm_name).all()
    subjects = Subject.query.filter_by(study_id=study_id).order_by(Subject.external_subject_code).all()
    
    if study_settings and study_settings.allowed_biomarkers:
        allowed_biomarker_ids = [b.biomarker_id for b in study_settings.allowed_biomarkers]
        
        available_biomarkers = db.session.query(
            BiomarkerType
        ).join(
            StudyRecordingBiomarker, BiomarkerType.biomarker_id == StudyRecordingBiomarker.biomarker_id
        ).filter(
            StudyRecordingBiomarker.study_id == study_id,
            BiomarkerType.biomarker_id.in_(allowed_biomarker_ids)
        ).distinct().order_by(BiomarkerType.biomarker_name).all()
    else:
        available_biomarkers = db.session.query(
            BiomarkerType
        ).join(
            StudyRecordingBiomarker, BiomarkerType.biomarker_id == StudyRecordingBiomarker.biomarker_id
        ).filter(
            StudyRecordingBiomarker.study_id == study_id
        ).distinct().order_by(BiomarkerType.biomarker_name).all()


    return render_template(
        'analytics.html', 
        study=study, 
        arms=arms, 
        subjects=subjects, 
        biomarkers=available_biomarkers,
        ai_available=ai_available
    )


def _build_categorical_plot(df, title, x_label, y_label):
    """
    Helper function to create a horizontal bar chart from a dataframe.
    Expects df with columns: 'Category' (y-axis) and 'Count' (x-axis).
    """
    df = df.sort_values('Count', ascending=True)
    source = ColumnDataSource(df)
    factors = df['Category'].tolist()
    
    plot_height = max(250, len(factors) * 35) 
    
    p = figure(
        title=title,
        y_range=FactorRange(factors=factors),
        height=plot_height, 
        sizing_mode='stretch_width',
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=[(y_label, "@Category"), (x_label, "@Count")]
    )
    
    p.hbar(y='Category', right='Count', source=source, height=0.8, color="#20519c")
    
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label

    labels = LabelSet(x='Count', y='Category', text='Count',
                      x_offset=5, y_offset=-7, 
                      source=source, text_font_size="9pt", text_color="#555555")
    p.add_layout(labels)
    
    plot_json = json.dumps(json_item(p, "categorical-bar-plot"))
    return jsonify({'status': 'success', 'type': 'plot', 'data': plot_json})


@app.route('/api/analytics/run/<study_id>', methods=['POST'])
@login_required
def run_analysis(study_id):
    """API endpoint to run analysis and return JSON for plots or stats."""
    _require_study_membership(study_id)
    try:
        study_settings = StudySettings.query.get(study_id)
        BOKEH_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided.'}), 400

        analysis_type = data.get('analysis_type')
        if not analysis_type:
             return jsonify({'status': 'error', 'message': 'Analysis Type is required.'}), 400

        single_biomarker_types = ['cohort_comparison', 'subject_over_time', 'distribution']
        multi_biomarker_types = ['correlation_matrix']
        
        comprehensive_types = ['comprehensive_correlation'] 
        
        automated_biomarker_types = ['automated_analysis']
        categorical_types = ['demographics_summary', 'event_frequency', 'medication_frequency']
        report_types = ['cohort_distribution_report']
        
        biomarker_ids = data.get('biomarker_ids', [])

        if analysis_type in single_biomarker_types:
            if len(biomarker_ids) != 1:
                return jsonify({'status': 'error', 'message': 'This analysis type requires exactly one biomarker.'}), 400
        
        elif analysis_type in multi_biomarker_types:
             if len(biomarker_ids) < 2:
                return jsonify({'status': 'error', 'message': 'This analysis type requires at least two biomarkers.'}), 400
        
        elif analysis_type in comprehensive_types:
             pass

        elif analysis_type in automated_biomarker_types:
             pass
        
        elif analysis_type in categorical_types:
             pass
        
        elif analysis_type in report_types:
            pass

        else:
            return jsonify({'status': 'error', 'message': 'Invalid analysis type specified.'}), 400

        if analysis_type in categorical_types:
            filter_arm_ids = data.get('filter_arm_ids', [])

            base_query = db.session.query(Subject)
            if filter_arm_ids:
                base_query = base_query.filter(Subject.arm_id.in_(filter_arm_ids))
            
            subjects_in_scope = base_query.filter(Subject.study_id == study_id).all()
            
            if not subjects_in_scope:
                return jsonify({'status': 'error', 'message': 'No subjects found matching the selected filters.'}), 404

            subject_ids_in_scope = [s.subject_id for s in subjects_in_scope]

            if analysis_type == 'demographics_summary':
                field = data.get('demographics_field')
                if not field:
                    return jsonify({'status': 'error', 'message': 'Please select a demographic field.'}), 400
                if field not in _ALLOWED_SUBJECT_FIELDS:
                    return jsonify({'status': 'error', 'message': 'Invalid demographic field.'}), 400

                results = db.session.query(
                    getattr(Subject, field),
                    func.count(Subject.subject_id)
                ).filter(
                    Subject.subject_id.in_(subject_ids_in_scope),
                    getattr(Subject, field) != None,
                    getattr(Subject, field) != ''
                ).group_by(
                    getattr(Subject, field)
                ).order_by(
                    func.count(Subject.subject_id).desc()
                ).all()

                if not results:
                     return jsonify({'status': 'error', 'message': f'No data found for demographic field "{field}".'}), 404

                df = pd.DataFrame(results, columns=['Category', 'Count'])
                df['Category'] = df['Category'].fillna('N/A')
                title = f"Demographics Summary: {field.replace('_', ' ').title()}"
                return _build_categorical_plot(df, title, "Number of Subjects", "Category")

            elif analysis_type == 'event_frequency':
                event_type = data.get('event_type')
                if not event_type:
                    return jsonify({'status': 'error', 'message': 'Please select an event type.'}), 400

                if event_type == 'Adverse Event':
                    query_model = SubjectAdverseEvent
                    field_to_group = SubjectAdverseEvent.meddra_term
                    title = "Adverse Event Frequency (Top 20)"
                else:
                    query_model = SubjectSymptom
                    field_to_group = SubjectSymptom.meddra_term
                    title = "Symptom Frequency (Top 20)"
                
                results = db.session.query(
                    field_to_group,
                    func.count(query_model.recording_id)
                ).filter(
                    query_model.subject_id.in_(subject_ids_in_scope)
                ).group_by(
                    field_to_group
                ).order_by(
                    func.count(query_model.recording_id).desc()
                ).limit(20).all() # Limit to top 20 for readability

                if not results:
                     return jsonify({'status': 'error', 'message': f'No {event_type.lower()} data found for the selected subjects.'}), 404
                
                df = pd.DataFrame(results, columns=['Category', 'Count'])
                df['Category'] = df['Category'].fillna('N/A')
                return _build_categorical_plot(df, title, "Total Occurrences", "MedDRA Term")

            elif analysis_type == 'medication_frequency':
                results = db.session.query(
                    SubjectMedicationTaken.medication_name,
                    func.count(SubjectMedicationTaken.recording_id)
                ).filter(
                    SubjectMedicationTaken.subject_id.in_(subject_ids_in_scope)
                ).group_by(
                    SubjectMedicationTaken.medication_name
                ).order_by(
                    func.count(SubjectMedicationTaken.recording_id).desc()
                ).limit(20).all() # Limit to top 20

                if not results:
                     return jsonify({'status': 'error', 'message': 'No concomitant medication data found for the selected subjects.'}), 404
                
                df = pd.DataFrame(results, columns=['Category', 'Count'])
                return _build_categorical_plot(df, "Concomitant Medication Frequency (Top 20)", "Total Occurrences", "Medication")


        elif analysis_type == 'cohort_distribution_report':

            # 1. Get all arms for the study
            arms = db.session.query(
                StudyArm.arm_id, 
                StudyArm.arm_name
            ).filter(StudyArm.study_id == study_id).order_by(StudyArm.arm_name).all()

            if not arms or len(arms) < 1:
                return jsonify({'status': 'error', 'message': 'This report requires at least one arm.'}), 404

            arm_ids = [c.arm_id for c in arms]
            arm_names_map = {c.arm_id: c.arm_name for c in arms}
            arm_names_list = [c.arm_name for c in arms]

            results_html = "<h3>Cohort Distribution Report</h3>"
            results_html += f"<p>This report compares the distribution of baseline characteristics for <strong>{len(arms)}</strong> arm(s).</p>"

            # 2. Get all subjects in scope
            demographic_fields = ['gender', 'ethnicity', 'race', 'handedness', 'smoking_status', 'alcohol_intake']
            subjects_df = pd.read_sql(
                db.session.query(Subject.subject_id, Subject.arm_id, *[getattr(Subject, f) for f in demographic_fields])
                .filter(Subject.study_id == study_id, Subject.arm_id.in_(arm_ids))
                .statement,
                db.engine
            )

            if subjects_df.empty:
                return jsonify({'status': 'error', 'message': 'No subjects found in any of the study arms.'}), 404

            subjects_df['arm_name'] = subjects_df['arm_id'].map(arm_names_map)
            subject_ids_in_scope = subjects_df['subject_id'].tolist()

            results_html += "<h4>1. Demographic Distribution</h4>"

            for field in demographic_fields:
                results_html += f"<h5>{field.replace('_', ' ').title()}</h5>"

                # Use a copy to avoid SettingWithCopyWarning
                field_df = subjects_df[['arm_name', field]].copy()
                field_df[field] = field_df[field].fillna('N/A').astype(str)

                count_pivot = pd.pivot_table(
                    field_df, 
                    index=field, 
                    columns='arm_name', 
                    aggfunc=len, 
                    fill_value=0,
                    margins=True,
                    margins_name='Total'
                )

                if count_pivot.empty or len(count_pivot) <= 1:
                    results_html += "<p class='text-muted'>No data found for this field.</p>"
                    continue

                percent_pivot = count_pivot.div(count_pivot.loc['Total'], axis=1).fillna(0) * 100
                results_html += _build_demographic_table(count_pivot, percent_pivot)

            diag_df = pd.read_sql(
                db.session.query(
                    Subject.arm_id, 
                    SubjectDiagnosis.diagnosis_description
                ).join(
                    Subject, Subject.subject_id == SubjectDiagnosis.subject_id
                ).filter(
                    Subject.subject_id.in_(subject_ids_in_scope)
                ).statement,
                db.engine
            )

            results_html += "<h4>2. Diagnosis History Distribution (Top 10)</h4>"
            if diag_df.empty:
                results_html += "<p class='text-muted'>No diagnosis history found for subjects in these arms.</p>"
            else:
                diag_df['arm_name'] = diag_df['arm_id'].map(arm_names_map)

                top_10_diags = diag_df['diagnosis_description'].value_counts().nlargest(10).index.tolist()
                diag_df_filtered = diag_df[diag_df['diagnosis_description'].isin(top_10_diags)]

                diag_pivot = pd.pivot_table(
                    diag_df_filtered,
                    index='diagnosis_description',
                    columns='arm_name',
                    aggfunc=len,
                    fill_value=0
                )

                # Re-order/add columns to match arm list
                diag_pivot = diag_pivot.reindex(columns=arm_names_list, fill_value=0)

                arm_totals = subjects_df.groupby('arm_name').size().reindex(arm_names_list, fill_value=0)
                diag_pivot['Total'] = diag_pivot.sum(axis=1)

                arm_totals['Total'] = arm_totals.sum()
                diag_pivot.loc['Total Subjects (N)'] = arm_totals

                percent_pivot = diag_pivot.div(diag_pivot.loc['Total Subjects (N)'], axis=1).fillna(0) * 100
                results_html += _build_distribution_table(diag_pivot, percent_pivot, "Diagnosis", arm_names_list)

            med_df = pd.read_sql(
                db.session.query(
                    Subject.arm_id, 
                    SubjectMedication.medication_name
                ).join(
                    Subject, Subject.subject_id == SubjectMedication.subject_id
                ).filter(
                    Subject.subject_id.in_(subject_ids_in_scope)
                ).statement,
                db.engine
            )

            results_html += "<h4>3. Medication History Distribution (Top 10)</h4>"
            if med_df.empty:
                results_html += "<p class='text-muted'>No medication history found for subjects in these arms.</p>"
            else:
                med_df['arm_name'] = med_df['arm_id'].map(arm_names_map)

                top_10_meds = med_df['medication_name'].value_counts().nlargest(10).index.tolist()
                med_df_filtered = med_df[med_df['medication_name'].isin(top_10_meds)]

                med_pivot = pd.pivot_table(
                    med_df_filtered,
                    index='medication_name',
                    columns='arm_name',
                    aggfunc=len,
                    fill_value=0
                )

                med_pivot = med_pivot.reindex(columns=arm_names_list, fill_value=0)

                arm_totals = subjects_df.groupby('arm_name').size().reindex(arm_names_list, fill_value=0)
                med_pivot['Total'] = med_pivot.sum(axis=1)

                arm_totals['Total'] = arm_totals.sum()
                med_pivot.loc['Total Subjects (N)'] = arm_totals

                percent_pivot = med_pivot.div(med_pivot.loc['Total Subjects (N)'], axis=1).fillna(0) * 100
                results_html += _build_distribution_table(med_pivot, percent_pivot, "Medication", arm_names_list)

            return jsonify({'status': 'success', 'type': 'stats', 'data': results_html})

        elif analysis_type in comprehensive_types:
            if analysis_type == 'comprehensive_correlation':
                
                arm_ids = data.get('correlation_arm_ids', [])
                
                # Get Subjects in scope (Demographics)
                base_subject_query = db.session.query(
                    Subject.subject_id,
                    Subject.external_subject_code,
                    Subject.gender,
                    Subject.date_of_birth,
                    Subject.ethnicity,
                    Subject.race,
                    Subject.height_cm,
                    Subject.weight_kg,
                    StudyArm.arm_name
                ).outerjoin(
                    StudyArm, Subject.arm_id == StudyArm.arm_id
                ).filter(Subject.study_id == study_id)
                
                if arm_ids:
                    base_subject_query = base_subject_query.filter(Subject.arm_id.in_(arm_ids))
                
                df_subjects = pd.read_sql(base_subject_query.statement, db.engine)
                
                if df_subjects.empty:
                    return jsonify({'status': 'error', 'message': 'No subjects found matching the selected arms.'}), 404

                # Add 'age' column
                if 'date_of_birth' in df_subjects.columns:
                    today = date.today()
                    df_subjects['age'] = df_subjects['date_of_birth'].apply(
                        lambda dob: (today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))) if pd.notnull(dob) else None
                    )
                
                df_subjects = df_subjects.drop(columns=['date_of_birth'], errors='ignore')
                subject_ids_in_scope = df_subjects['subject_id'].tolist()
                
                # Get Biomarker Data (Mean per subject)
                biomarker_query = db.session.query(
                    StudyRecordingBiomarker.subject_id,
                    BiomarkerType.biomarker_name,
                    StudyRecordingBiomarker.biomarker_value
                ).join(
                    BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id
                ).filter(
                    StudyRecordingBiomarker.subject_id.in_(subject_ids_in_scope)
                )
                df_biomarkers = pd.read_sql(biomarker_query.statement, db.engine)
                
                df_biomarkers_pivot = pd.DataFrame(index=df_subjects.set_index('subject_id').index)
                if not df_biomarkers.empty:
                    df_biomarkers['biomarker_value'] = pd.to_numeric(df_biomarkers['biomarker_value'], errors='coerce')
                    df_biomarkers_avg = df_biomarkers.dropna(subset=['biomarker_value']).groupby(['subject_id', 'biomarker_name'])['biomarker_value'].mean().reset_index()
                    
                    if not df_biomarkers_avg.empty:
                        df_biomarkers_pivot = df_biomarkers_avg.pivot(index='subject_id', columns='biomarker_name', values='biomarker_value')
                        df_biomarkers_pivot = df_biomarkers_pivot.add_prefix('bio_')
                
                # Get Symptom Data (Count per subject)
                symptom_query = db.session.query(
                    SubjectSymptom.subject_id,
                    SubjectSymptom.meddra_term
                ).filter(
                    SubjectSymptom.subject_id.in_(subject_ids_in_scope)
                )
                df_symptoms = pd.read_sql(symptom_query.statement, db.engine)
                
                df_symptoms_pivot = pd.DataFrame(index=df_subjects.set_index('subject_id').index)
                if not df_symptoms.empty:
                    # Handle potential empty meddra_term
                    df_symptoms['meddra_term'] = df_symptoms['meddra_term'].fillna('Unknown_Symptom')
                    df_symptoms_count = df_symptoms.groupby(['subject_id', 'meddra_term']).size().unstack(fill_value=0)
                    df_symptoms_pivot = df_symptoms_count.add_prefix('sym_')
                
                # Get AE Data (Count per subject)
                ae_query = db.session.query(
                    SubjectAdverseEvent.subject_id,
                    SubjectAdverseEvent.meddra_term
                ).filter(
                    SubjectAdverseEvent.subject_id.in_(subject_ids_in_scope)
                )
                df_aes = pd.read_sql(ae_query.statement, db.engine)
                
                df_aes_pivot = pd.DataFrame(index=df_subjects.set_index('subject_id').index)
                if not df_aes.empty:
                    # Handle potential empty meddra_term
                    df_aes['meddra_term'] = df_aes['meddra_term'].fillna('Unknown_AE')
                    df_aes_count = df_aes.groupby(['subject_id', 'meddra_term']).size().unstack(fill_value=0)
                    df_aes_pivot = df_aes_count.add_prefix('ae_')
                
                # Get Medication Data (Count per subject)
                med_query = db.session.query(
                    SubjectMedicationTaken.subject_id,
                    SubjectMedicationTaken.medication_name
                ).filter(
                    SubjectMedicationTaken.subject_id.in_(subject_ids_in_scope)
                )
                df_meds = pd.read_sql(med_query.statement, db.engine)
                
                df_meds_pivot = pd.DataFrame(index=df_subjects.set_index('subject_id').index)
                if not df_meds.empty:
                    # Handle potential empty medication_name
                    df_meds['medication_name'] = df_meds['medication_name'].fillna('Unknown_Medication')
                    df_meds_count = df_meds.groupby(['subject_id', 'medication_name']).size().unstack(fill_value=0)
                    df_meds_pivot = df_meds_count.add_prefix('med_')

                # Combine all DataFrames
                df_main = df_subjects.set_index('subject_id')
                df_main = df_main.join(df_biomarkers_pivot, how='left')
                df_main = df_main.join(df_symptoms_pivot, how='left')
                df_main = df_main.join(df_aes_pivot, how='left')
                df_main = df_main.join(df_meds_pivot, how='left')
                
                # Pre-process for correlation
                categorical_cols = ['gender', 'ethnicity', 'race', 'arm_name']
                df_corr = pd.get_dummies(df_main, columns=categorical_cols, drop_first=True, dummy_na=True)
                df_corr = df_corr.drop(columns=['external_subject_code'], errors='ignore')

                count_cols = [col for col in df_corr.columns if col.startswith('sym_') or col.startswith('ae_') or col.startswith('med_')]
                df_corr[count_cols] = df_corr[count_cols].fillna(0)
                
                numeric_cols = ['age', 'height_cm', 'weight_kg'] + [col for col in df_corr.columns if col.startswith('bio_')]
                for col in numeric_cols:
                    if col in df_corr.columns:
                        try:
                            # Ensure column is numeric before median
                            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
                            median_val = df_corr[col].median()
                            if pd.notna(median_val):
                                df_corr[col] = df_corr[col].fillna(median_val)
                        except Exception as e:
                            app.logger.warning(f"Could not process median for column {col}: {e}")

                # Drop columns with no variance
                df_corr = df_corr.loc[:, df_corr.nunique(dropna=False) > 1] # dropna=False to catch cols with 1 value + NaN
                # Drop cols that are all NaN
                df_corr = df_corr.dropna(axis=1, how='all')

                if df_corr.shape[1] < 2:
                    return jsonify({'status': 'error', 'message': 'Not enough variable data found to perform a correlation. Try including more subjects or arms.'}), 404
                
                # Calculate Correlation Matrix (Spearman)
                corr_matrix = df_corr.corr(method='spearman')
                
                # Find interesting correlations
                corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_series = corr_matrix_upper.stack().reset_index()
                corr_series.columns = ['Variable 1', 'Variable 2', 'Correlation']
                corr_series['Abs_Correlation'] = corr_series['Correlation'].abs()
                
                # Filter out very weak correlations to make report more meaningful
                strong_corrs = corr_series[corr_series['Abs_Correlation'] > 0.1].nlargest(10, 'Abs_Correlation') # Show top 10 with rho > 0.1
                results_html = "<h3>Comprehensive Correlation Analysis (Spearman's Rho)</h3>"
                results_html += f"<p>This analysis correlates subject demographics, mean biomarker values, and event counts (Symptoms, AEs, Medications) for <strong>{len(df_main)}</strong> subjects. Only correlations with an absolute rho > 0.1 are considered, and the top 10 strongest are shown.</p>"
                
                if strong_corrs.empty:
                    results_html += "<div class='alert alert-warning'>No notable correlations (absolute rho > 0.1) were found with the available data.</div>"
                else:
                    results_html += "<div class='table-responsive'>"
                    results_html += "<table class='table table-sm table-striped table-bordered'>"
                    results_html += """
                        <thead class='thead-light'>
                            <tr>
                                <th>Variable 1</th>
                                <th>Variable 2</th>
                                <th>Spearman Correlation (rho)</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for _, row in strong_corrs.iterrows():
                        v1_formatted = _format_variable_name(row['Variable 1'])
                        v2_formatted = _format_variable_name(row['Variable 2'])
                        results_html += f"""
                            <tr>
                                <td>{v1_formatted}</td>
                                <td>{v2_formatted}</td>
                                <td>{row['Correlation']:.3f}</td>
                            </tr>
                        """
                    results_html += "</tbody></table></div>"

                return jsonify({'status': 'success', 'type': 'stats', 'data': results_html})

        elif analysis_type in automated_biomarker_types:
            if not study_settings or not study_settings.ai_enabled:
                return jsonify({'status': 'error', 'message': 'AI analysis is not enabled for this study. Please enable it in the study settings.'}), 400
            
            api_key = study_settings.ai_api_key
            if not api_key:
                return jsonify({'status': 'error', 'message': 'AI analysis is enabled, but no API key is configured for this study.'}), 500
            
            try:
                client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Failed to initialize AI service: {str(e)}'}), 500

            arm_ids = data.get('correlation_arm_ids', [])
            base_subject_query = db.session.query(Subject).filter(Subject.study_id == study_id)
            if arm_ids:
                base_subject_query = base_subject_query.filter(Subject.arm_id.in_(arm_ids))
            
            subjects_in_scope = base_subject_query.all()
            if not subjects_in_scope:
                return jsonify({'status': 'error', 'message': 'No subjects found matching the selected arms.'}), 404
            
            subject_ids_in_scope = [s.subject_id for s in subjects_in_scope]
            app.logger.info(f"AI Analysis: Found {len(subject_ids_in_scope)} subjects in scope.")

            df_demographics = pd.read_sql(base_subject_query.statement, db.engine)
            df_demographics = df_demographics.drop(columns=[
                'subject_id', 'study_id', 'arm_id', 'first_name', 'last_name', 
                'city', 'state_province', 'country', 'postal_code', 
                'consent_date', 'withdrawal_date'
            ], errors='ignore')

            biomarker_query = db.session.query(
                Subject.external_subject_code,
                StudyArm.arm_name,
                BiomarkerType.biomarker_name,
                StudyRecordingBiomarker.biomarker_value
            ).join(
                Subject, StudyRecordingBiomarker.subject_id == Subject.subject_id
            ).join(
                BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id
            ).outerjoin(
                StudyArm, Subject.arm_id == StudyArm.arm_id
            ).filter(
                StudyRecordingBiomarker.subject_id.in_(subject_ids_in_scope)
            )
            df_biomarkers = pd.read_sql(biomarker_query.statement, db.engine)
            
            ae_query = db.session.query(
                SubjectAdverseEvent
            ).filter(
                SubjectAdverseEvent.subject_id.in_(subject_ids_in_scope)
            )
            df_aes = pd.read_sql(ae_query.statement, db.engine)

            symptom_query = db.session.query(
                SubjectSymptom
            ).filter(
                SubjectSymptom.subject_id.in_(subject_ids_in_scope)
            )
            df_symptoms = pd.read_sql(symptom_query.statement, db.engine)

            med_query = db.session.query(
                SubjectMedicationTaken
            ).filter(
                SubjectMedicationTaken.subject_id.in_(subject_ids_in_scope)
            )
            df_meds = pd.read_sql(med_query.statement, db.engine)

            data_prompt_segment = "--- START DATASET (SUMMARIES) ---\n\n"
            
            if not df_demographics.empty:
                try:
                    data_prompt_segment += "--- DEMOGRAPHICS (NUMERICAL SUMMARY) ---\n"
                    data_prompt_segment += df_demographics.describe().to_csv() + "\n\n"
                    data_prompt_segment += "--- DEMOGRAPHICS (CATEGORICAL SUMMARY) ---\n"
                    data_prompt_segment += df_demographics.describe(include='object').to_csv() + "\n\n"
                except Exception:  # suppress describe() errors on unexpected column types
                    pass

            if not df_biomarkers.empty:
                try:
                    df_biomarkers['biomarker_value'] = pd.to_numeric(df_biomarkers['biomarker_value'], errors='coerce')
                    df_biomarkers_pivot = df_biomarkers.pivot_table(
                        index=['external_subject_code', 'arm_name'],
                        columns='biomarker_name',
                        values='biomarker_value',
                        aggfunc='mean'
                    ).reset_index()
                    data_prompt_segment += "--- BIOMARKER SUMMARY (MEAN VALUE PER SUBJECT) ---\n"
                    data_prompt_segment += df_biomarkers_pivot.to_csv(index=False) + "\n\n"
                except Exception as e:
                    app.logger.warning(f"AI Analysis: Could not pivot biomarker data: {e}")

            if not df_aes.empty:
                data_prompt_segment += "--- ADVERSE EVENT FREQUENCY ---\n"
                data_prompt_segment += df_aes['meddra_term'].value_counts().to_csv() + "\n\n"

            if not df_symptoms.empty:
                data_prompt_segment += "--- SYMPTOM FREQUENCY ---\n"
                data_prompt_segment += df_symptoms['meddra_term'].value_counts().to_csv() + "\n\n"
            
            if not df_meds.empty:
                data_prompt_segment += "--- CONCOMITANT MEDICATION FREQUENCY ---\n"
                data_prompt_segment += df_meds['medication_name'].value_counts().to_csv() + "\n\n"

            data_prompt_segment += "--- END DATASET ---\n"
            
            if data_prompt_segment == "--- START DATASET (SUMMARIES) ---\n\n--- END DATASET ---\n":
                return jsonify({'status': 'error', 'message': 'No data (demographics, biomarkers, AEs, etc.) found for the selected subjects.'}), 404

            MAX_CSV_LENGTH = 10000 
            if len(data_prompt_segment) > MAX_CSV_LENGTH:
                return jsonify({'status': 'error', 'message': f'Selected data is too large ({len(data_prompt_segment)} characters) for automated analysis. Please filter by arms to reduce data size.'}), 400
            
            vector_store_exists_record = db.session.query(StudyKnowledgeVector.vector_id)\
                .join(StudyKnowledge, StudyKnowledge.knowledge_id == StudyKnowledgeVector.knowledge_id)\
                .filter(StudyKnowledge.study_id == study_id)\
                .first()
            vector_store_exists = vector_store_exists_record is not None

            rag_prompt_addition = ""
            if vector_store_exists:
                app.logger.info(f"Vector store found. Attempting RAG context for study {study_id}.")

                rag_query_text = "General study analysis"
                if arm_ids:
                    selected_arms = StudyArm.query.filter(StudyArm.arm_id.in_(arm_ids)).all()
                    arm_names = [c.arm_name for c in selected_arms]
                    if arm_names:
                         rag_query_text += f" comparing arms: {', '.join(arm_names)}"

                knowledge_context = _get_rag_context(study_id, rag_query_text, max_context_items=5)
                
                if knowledge_context:
                    app.logger.info(f"Using RAG context for study {study_id}.")
                    rag_prompt_addition = f"""
                    --- CONTEXT FROM UPLOADED DOCUMENTS ---
                    The user has provided the following documents which may be relevant. Use this context to inform your analysis.
                    {knowledge_context}
                    --- END CONTEXT ---
                    """
                else:
                    app.logger.info(f"Vector store exists, but no relevant RAG context found for study {study_id}.")
            else:
                app.logger.info(f"No vector store (studies_knowledge_vector) found for study {study_id}. Skipping RAG.")
            
            system_prompt = "You are a medical researcher and biostatistician. Your task is to analyze clinical trial data and present your findings in a clear, well-structured HTML report."

            user_prompt = f"""
            Analyze the following clinical trial data summaries. The data includes demographics, mean biomarker values per subject, adverse event frequencies, symptom frequencies, and medication frequencies. Provide a full summary of potential interesting findings.
            
            Look for:
            - Obvious trends or issues in the demographic data.
            - Differences in biomarker values, AEs, or symptoms between arms (if arm data is present).
            - Any notable correlations or co-occurrences (e.g., "subjects in Arm A had high X biomarker and also reported Y symptom frequently").
            - The most common adverse events, symptoms, and medications.
            
            {rag_prompt_addition}

            If you can find **relevant, published journal articles** that provide context or support for your findings, create a section titled `<h4>Published Journal Articles</h4>` and list their **full, valid citations** (e.g., Author(s). (Year). Title. Journal, Vol(Issue), pages. PMID or DOI).
            
            **Important:** Do *not* include this section or any placeholder citations if you cannot find any real, valid articles.

            Structure your entire response as a **self-contained HTML report fragment**. Use appropriate headers (e.g., `<h3>`, `<h4>`), paragraphs (`<p>`), lists (`<ul>`, `<li>`), and bold text (`<strong>`) for clarity. Do NOT include `<html>`, `<body>`, or `<!DOCTYPE html>` tags. Start directly with a header like `<h3>Automated Data Analysis Report</h3>`.

            Here is the dataset:
            {data_prompt_segment}
            """

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1, 
                    max_tokens=2500 
                )
                report_html = response.choices[0].message.content
                report_html = report_html.strip()
                if not report_html.startswith("<h"):
                    report_html = "<h3>Automated Data Analysis Report</h3>" + report_html
                return jsonify({'status': 'success', 'type': 'stats', 'data': report_html})
            except openai.APIError as e:
                    return jsonify({'status': 'error', 'message': f'AI service returned an error: {str(e)}'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'An unexpected error occurred while contacting the AI service: {str(e)}'}), 500

        else:
            base_query = db.session.query(
                Subject.external_subject_code,
                Subject.subject_id,
                StudyArm.arm_name,
                StudyArm.arm_id,
                StudyRecording.recording_datetime,
                StudyRecordingBiomarker.biomarker_id,
                StudyRecordingBiomarker.biomarker_value,
                BiomarkerType.biomarker_name
            ).join(
                StudyRecording, Subject.subject_id == StudyRecording.subject_id
            ).join(
                StudyRecordingBiomarker, StudyRecording.recording_id == StudyRecordingBiomarker.recording_id
            ).join(
                BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id
            ).outerjoin(
                StudyArm, Subject.arm_id == StudyArm.arm_id
            ).filter(
                Subject.study_id == study_id,
                StudyRecordingBiomarker.biomarker_id.in_(biomarker_ids)
            )
            df = pd.read_sql(base_query.statement, db.engine)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No matching biomarker data found for the selected biomarker(s).'}), 404
            
            df['arm_id'] = df['arm_id'].apply(lambda x: str(x) if pd.notna(x) else None)
            df['subject_id'] = df['subject_id'].apply(lambda x: str(x) if pd.notna(x) else None)
            df['biomarker_id'] = df['biomarker_id'].apply(lambda x: str(x) if pd.notna(x) else None)
            df['biomarker_value'] = pd.to_numeric(df['biomarker_value'], errors='coerce') 
            biomarker_name = df['biomarker_name'].iloc[0] if not df.empty else "Selected Biomarker"

            if df['biomarker_value'].isnull().all():
                return jsonify({'status': 'error', 'message': 'No valid numeric data found for the selected biomarker(s). All recorded values are null or non-numeric.'}), 404

            if analysis_type == 'cohort_comparison':
                cohort_ids = data.get('cohort_ids', [])
                test_type = data.get('cohort_test')

                if not cohort_ids:
                    return jsonify({'status': 'error', 'message': 'Please select at least one cohort.'}), 400

                df_filtered = df[df['arm_id'].isin(cohort_ids)].copy()
                df_filtered.dropna(subset=['biomarker_value'], inplace=True) 

                if df_filtered.empty:
                    return jsonify({'status': 'error', 'message': 'No valid data found for the selected cohorts.'}), 404

                unique_arms_with_data = df_filtered['arm_name'].nunique()
                
                if test_type == 'box_plot':
                    arms = sorted(df_filtered['arm_name'].unique().tolist())
                    grouped = df_filtered.groupby('arm_name')
                    q = grouped['biomarker_value'].quantile([0.25, 0.5, 0.75])
                    q = q.unstack().reset_index()
                    q.columns = ['arm_name', 'q1', 'q2', 'q3']
                    iqr = q['q3'] - q['q1']
                    q['upper'] = q['q3'] + 1.5 * iqr
                    q['lower'] = q['q1'] - 1.5 * iqr
                    min_vals = grouped['biomarker_value'].min()
                    max_vals = grouped['biomarker_value'].max()
                    q['upper'] = q.apply(lambda row: min(row['upper'], max_vals.get(row['arm_name'], row['upper'])), axis=1)
                    q['lower'] = q.apply(lambda row: max(row['lower'], min_vals.get(row['arm_name'], row['lower'])), axis=1)

                    source = ColumnDataSource(q)

                    p = figure(
                        x_range=arms, 
                        title=f'"{biomarker_name}" by Arm', 
                        tools=BOKEH_TOOLS,
                        height=400, 
                        sizing_mode='stretch_width'
                    )
                    p.segment(x0='arm_name', y0='upper', x1='arm_name', y1='q3', source=source, line_color="black")
                    p.segment(x0='arm_name', y0='lower', x1='arm_name', y1='q1', source=source, line_color="black")
                    p.vbar(x='arm_name', width=0.7, top='q3', bottom='q1', source=source, fill_color="#E08E79", line_color="black")
                    p.vbar(x='arm_name', width=0.7, top='q2', bottom='q2', source=source, line_color="black", line_width=2) # Median
                    all_outliers = pd.DataFrame(columns=['arm_name', 'biomarker_value'])
                    for arm in arms:
                        arm_data = df_filtered[df_filtered['arm_name'] == arm]
                        stats = q[q['arm_name'] == arm].iloc[0]
                        outliers = arm_data[
                            (arm_data['biomarker_value'] > stats['upper']) | 
                            (arm_data['biomarker_value'] < stats['lower'])
                        ]
                        all_outliers = pd.concat([all_outliers, outliers[['arm_name', 'biomarker_value']]])
                    
                    if not all_outliers.empty:
                        outlier_source = ColumnDataSource(all_outliers)
                        p.scatter(x='arm_name', y='biomarker_value', source=outlier_source, color='black', size=6, alpha=0.6)

                    p.xaxis.axis_label = 'Arm'
                    p.yaxis.axis_label = biomarker_name
                    p.xgrid.grid_line_color = None

                    plot_json = json.dumps(json_item(p, "arm-box-plot"))
                    return jsonify({'status': 'success', 'type': 'plot', 'data': plot_json})

                elif test_type == 'violin_plot':
                    arms = sorted(df_filtered['arm_name'].unique().tolist())
                    if len(arms) > 10:
                        colors = Category20[len(arms)] if len(arms) <= 20 else ['blue'] * len(arms)
                    else:
                        colors = Category10[max(3, len(arms))][:len(arms)]

                    p = figure(
                        title=f'"{biomarker_name}" by Arm (Density Plot)', 
                        tools=BOKEH_TOOLS, 
                        height=400, 
                        sizing_mode='stretch_width'
                    )

                    for i, arm in enumerate(arms):
                        arm_df = df_filtered[df_filtered['arm_name'] == arm]
                        data = arm_df['biomarker_value'].dropna()
                        if data.empty or len(data) < 2:
                            continue
                        
                        try:
                            density = gaussian_kde(data)
                            xs = np.linspace(data.min(), data.max(), 200)
                            ys = density(xs)
                            p.patch(xs, ys, fill_color=colors[i], fill_alpha=0.4, line_color=colors[i], line_width=2, legend_label=arm)
                        except (np.linalg.LinAlgError, ValueError):
                            app.logger.warning(f"Could not compute KDE for arm '{arm}'")
                            p.scatter(x=data, y=np.zeros_like(data), color=colors[i], legend_label=f"{arm} (insufficient variance for KDE)")

                    p.xaxis.axis_label = biomarker_name
                    p.yaxis.axis_label = 'Density'
                    if p.legend:
                        p.legend.click_policy = 'hide'

                    plot_json = json.dumps(json_item(p, "arm-density-plot"))
                    return jsonify({'status': 'success', 'type': 'plot', 'data': plot_json})

                elif test_type == 't_test':
                    if unique_arms_with_data != 2:
                        return jsonify({'status': 'error', 'message': f'T-Test requires exactly two arms with valid data. Found {unique_arms_with_data}.'}), 400

                    arms = [group['biomarker_value'].values for name, group in df_filtered.groupby('arm_name')]
                    arm_names = df_filtered['arm_name'].unique()

                    stat, p_value = ttest_ind(arms[0], arms[1], nan_policy='omit')

                    html = f"<h3>Independent T-Test: {arm_names[0]} vs {arm_names[1]}</h3>"
                    html += f"<p><strong>Biomarker:</strong> {biomarker_name}</p><hr>"
                    html += f"<p><strong>T-statistic:</strong> {stat:.4f}</p>"
                    html += f"<p><strong>P-value:</strong> {p_value:.4f}</p>"
                    html += "<div class='alert alert-info'>"
                    if p_value < 0.05:
                        html += "<strong>Result:</strong> The difference between the means of the two arms is statistically significant (p < 0.05)."
                    else:
                        html += "<strong>Result:</strong> There is no statistically significant difference between the means of the two arms (p >= 0.05)."
                    html += "</div>"
                    return jsonify({'status': 'success', 'type': 'stats', 'data': html})

                elif test_type == 'anova':
                    if unique_arms_with_data < 2:
                        return jsonify({'status': 'error', 'message': f'ANOVA requires at least two arms with valid data. Found {unique_arms_with_data}.'}), 400

                    arms_data = [group['biomarker_value'].values for name, group in df_filtered.groupby('arm_name')]
                    arm_names_used = df_filtered['arm_name'].unique()

                    stat, p_value = f_oneway(*arms_data)

                    html = "<h3>One-Way ANOVA</h3>"
                    html += f"<p><strong>Biomarker:</strong> {biomarker_name}</p>"
                    html += f"<p><strong>Arms Included:</strong> {', '.join(arm_names_used)}</p><hr>"
                    html += f"<p><strong>F-statistic:</strong> {stat:.4f}</p>"
                    html += f"<p><strong>P-value:</strong> {p_value:.4f}</p>"
                    html += "<div class='alert alert-info'>"
                    if p_value < 0.05:
                        html += "<strong>Result:</strong> There is a statistically significant difference between the means of at least two of the arms (p < 0.05)."
                    else:
                        html += "<strong>Result:</strong> There is no statistically significant difference between the means of the arms (p >= 0.05)."
                    html += "</div>"
                    return jsonify({'status': 'success', 'type': 'stats', 'data': html})

                elif test_type == 'mannwhitneyu':
                    if unique_arms_with_data != 2:
                        return jsonify({'status': 'error', 'message': f'Mann-Whitney U Test requires exactly two arms with valid data. Found {unique_arms_with_data}.'}), 400

                    arms = [group['biomarker_value'].values for name, group in df_filtered.groupby('arm_name')]
                    arm_names = df_filtered['arm_name'].unique()

                    g1_clean = arms[0][~np.isnan(arms[0])]
                    g2_clean = arms[1][~np.isnan(arms[1])]

                    if len(g1_clean) < 1 or len(g2_clean) < 1:
                            return jsonify({'status': 'error', 'message': 'Not enough valid data in one or both arms for Mann-Whitney U Test.'}), 400

                    try:
                        stat, p_value = mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')
                    except ValueError as e:
                            return jsonify({'status': 'error', 'message': f'Could not perform Mann-Whitney U test (e.g., all values identical?): {e}'}), 400

                    html = f"<h3>Mann-Whitney U Test: {arm_names[0]} vs {arm_names[1]}</h3>"
                    html += f"<p><strong>Biomarker:</strong> {biomarker_name}</p><hr>"
                    html += f"<p><strong>U statistic:</strong> {stat:.4f}</p>"
                    html += f"<p><strong>P-value:</strong> {p_value:.4f}</p>"
                    html += "<div class='alert alert-info'>"
                    if p_value < 0.05:
                        html += "<strong>Result:</strong> The distributions of the two arms are statistically significantly different (p < 0.05)."
                    else:
                        html += "<strong>Result:</strong> There is no statistically significant difference between the distributions of the two arms (p >= 0.05)."
                    html += "</div>"
                    return jsonify({'status': 'success', 'type': 'stats', 'data': html})

                elif test_type == 'kruskalwallis':
                    if unique_arms_with_data < 2:
                        return jsonify({'status': 'error', 'message': f'Kruskal-Wallis Test requires at least two arms with valid data. Found {unique_arms_with_data}.'}), 400

                    arms_data = [group['biomarker_value'].dropna().values
                                    for name, group in df_filtered.groupby('arm_name')
                                    if group['biomarker_value'].notna().any()]

                    if len(arms_data) < 2:
                            return jsonify({'status': 'error', 'message': 'Not enough distinct arms with valid (non-NaN) data to perform Kruskal-Wallis Test.'}), 400

                    arm_names_used = [name for name, group in df_filtered.groupby('arm_name') if group['biomarker_value'].notna().any()]

                    try:
                        stat, p_value = kruskal(*arms_data)
                    except ValueError as e:
                            return jsonify({'status': 'error', 'message': f'Could not perform Kruskal-Wallis test: {e}'}), 400

                    html = "<h3>Kruskal-Wallis H Test</h3>"
                    html += f"<p><strong>Biomarker:</strong> {biomarker_name}</p>"
                    html += f"<p><strong>Arms Included:</strong> {', '.join(arm_names_used)}</p><hr>"
                    html += f"<p><strong>H statistic:</strong> {stat:.4f}</p>"
                    html += f"<p><strong>P-value:</strong> {p_value:.4f}</p>"
                    html += "<div class='alert alert-info'>"
                    if p_value < 0.05:
                        html += "<strong>Result:</strong> There is a statistically significant difference between the distributions of at least two of the arms (p < 0.05)."
                    else:
                        html += "<strong>Result:</strong> There is no statistically significant difference between the distributions of the arms (p >= 0.05)."
                    html += "</div>"
                    return jsonify({'status': 'success', 'type': 'stats', 'data': html})

            elif analysis_type == 'subject_over_time':
                subject_codes = data.get('subject_codes', [])
                plot_type = data.get('subject_plot')

                if not subject_codes:
                    return jsonify({'status': 'error', 'message': 'Please select at least one subject.'}), 400

                df_filtered = df[df['external_subject_code'].isin(subject_codes)].sort_values('recording_datetime')

                if df_filtered.empty:
                    return jsonify({'status': 'error', 'message': 'No data found for the selected subjects.'}), 404

                if df_filtered['biomarker_value'].isnull().all():
                    return jsonify({'status': 'error', 'message': f'No non-empty data found for {biomarker_name} for the selected subjects.'}), 404

                title = f'"{biomarker_name}" Over Time'
                
                p = figure(
                    title=title, 
                    x_axis_type='datetime', 
                    height=400, 
                    sizing_mode='stretch_width', 
                    tools=BOKEH_TOOLS
                )
                
                subjects = df_filtered['external_subject_code'].unique().tolist()
                if len(subjects) > 10:
                    colors = Category20[min(20, len(subjects))]
                else:
                    colors = Category10[max(3, len(subjects))]

                for i, subject in enumerate(subjects):
                    subject_df = df_filtered[df_filtered['external_subject_code'] == subject].sort_values('recording_datetime')
                    if subject_df.empty:
                        continue
                    
                    source = ColumnDataSource(subject_df)
                    color = colors[i % len(colors)]
                    
                    if plot_type == 'line_plot':
                        p.line(x='recording_datetime', y='biomarker_value', source=source, legend_label=subject, color=color, line_width=2)
                        p.scatter(x='recording_datetime', y='biomarker_value', source=source, legend_label=subject, color=color, size=5)
                    else: # scatter_plot
                        p.scatter(x='recording_datetime', y='biomarker_value', source=source, legend_label=subject, color=color, size=5)

                p.xaxis.axis_label = 'Date'
                p.yaxis.axis_label = biomarker_name
                if p.legend:
                    p.legend.click_policy = 'hide'

                p.add_tools(HoverTool(
                    tooltips=[
                        ('Subject', '@external_subject_code'),
                        ('Date', '@recording_datetime{%F %T}'),
                        (biomarker_name, '@biomarker_value')
                    ],
                    formatters={'@recording_datetime': 'datetime'}
                ))

                plot_json = json.dumps(json_item(p, "subject-time-plot"))
                return jsonify({'status': 'success', 'type': 'plot', 'data': plot_json})

            elif analysis_type == 'distribution':
                plot_type = data.get('distribution_plot')
                df_filtered = df.dropna(subset=['biomarker_value'])

                if df_filtered.empty:
                    return jsonify({'status': 'error', 'message': f'No valid data found for {biomarker_name} distribution.'}), 404

                data_clean = df_filtered['biomarker_value']

                if plot_type == 'histogram':
                    title = f'Distribution of "{biomarker_name}"'
                    hist, edges = np.histogram(data_clean, bins='auto')
                    
                    p = figure(
                        title=title, 
                        tools=BOKEH_TOOLS, 
                        height=400, 
                        sizing_mode='stretch_width', 
                        background_fill_color="#fafafa"
                    )
                    p.quad(
                        top=hist, 
                        bottom=0, 
                        left=edges[:-1], 
                        right=edges[1:], 
                        fill_color="#1f77b4", 
                        line_color="white", 
                        alpha=0.75
                    )
                    p.xaxis.axis_label = biomarker_name
                    p.yaxis.axis_label = 'Frequency'
                
                else:
                    title = f'Density Plot of "{biomarker_name}"'
                    p = figure(
                        title=title, 
                        tools=BOKEH_TOOLS, 
                        height=400, 
                        sizing_mode='stretch_width'
                    )
                    
                    try:
                        density = gaussian_kde(data_clean)
                        xs = np.linspace(data_clean.min(), data_clean.max(), 200)
                        ys = density(xs)
                        p.patch(xs, ys, color="#1f77b4", fill_alpha=0.6, line_color="black")
                    except (np.linalg.LinAlgError, ValueError):
                        return jsonify({'status': 'error', 'message': 'Could not compute density plot (insufficient variance in data).'}), 400
                    
                    p.xaxis.axis_label = biomarker_name
                    p.yaxis.axis_label = 'Density'

                plot_json = json.dumps(json_item(p, "distribution-plot"))
                return jsonify({'status': 'success', 'type': 'plot', 'data': plot_json})

            elif analysis_type == 'correlation_matrix':
                arm_ids = data.get('correlation_arm_ids', [])

                df_filtered = df
                if arm_ids:
                    df_filtered = df[df['arm_id'].isin(arm_ids)]

                if df_filtered.empty:
                        return jsonify({'status': 'error', 'message': 'No data found for the selected arms (or no arms selected and no data exists).'}), 404

                df_pivot_avg = df_filtered.dropna(subset=['biomarker_value']).groupby(['subject_id', 'biomarker_name'])['biomarker_value'].mean().reset_index()
                df_pivot = df_pivot_avg.pivot(index='subject_id', columns='biomarker_name', values='biomarker_value')

                if df_pivot.shape[0] < 2 or df_pivot.shape[1] < 2:
                        return jsonify({'status': 'error', 'message': f'Not enough subjects ({df_pivot.shape[0]}) or biomarkers ({df_pivot.shape[1]}) with comparable data to calculate correlation.'}), 404

                results_html = "<h3>Pairwise Biomarker Correlations</h3><hr>"
                biomarkers = df_pivot.columns.astype(str).tolist()
                results_list = []

                for i in range(len(biomarkers)):
                    for j in range(i + 1, len(biomarkers)):
                        b1_name = biomarkers[i]
                        b2_name = biomarkers[j]
                        pair_data = df_pivot[[b1_name, b2_name]].dropna()

                        if len(pair_data) < 2:
                            results_list.append({
                                'b1': b1_name, 'b2': b2_name, 'pearson_r': 'N/A', 'pearson_p': 'N/A',
                                'spearman_rho': 'N/A', 'spearman_p': 'N/A', 'n': len(pair_data), 'error': 'Too few non-NaN pairs'
                            })
                            continue

                        try:
                            pearson_corr, pearson_p_value = pearsonr(pair_data[b1_name], pair_data[b2_name])
                        except ValueError:
                            pearson_corr, pearson_p_value = float('nan'), float('nan')

                        try:
                            spearman_corr, spearman_p_value = spearmanr(pair_data[b1_name], pair_data[b2_name])
                        except ValueError:
                                spearman_corr, spearman_p_value = float('nan'), float('nan')

                        results_list.append({
                            'b1': b1_name, 'b2': b2_name, 'pearson_r': f"{pearson_corr:.3f}", 'pearson_p': f"{pearson_p_value:.3f}",
                            'spearman_rho': f"{spearman_corr:.3f}", 'spearman_p': f"{spearman_p_value:.3f}", 'n': len(pair_data), 'error': None
                        })

                if not results_list:
                        results_html += "<p>No biomarker pairs found with sufficient data for correlation.</p>"
                else:
                    results_html += "<div class='table-responsive'>"
                    results_html += "<table class='table table-sm table-striped table-bordered'>"
                    results_html += """
                        <thead class='thead-light'>
                            <tr>
                                <th>Biomarker 1</th>
                                <th>Biomarker 2</th>
                                <th>N (Pairs)</th>
                                <th>Pearson's r</th>
                                <th>Pearson P-value</th>
                                <th>Spearman's rho</th>
                                <th>Spearman P-value</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    for res in results_list:
                        p_pearson_class = 'text-danger fw-bold' if res['pearson_p'] != 'N/A' and not pd.isna(float(res['pearson_p'])) and float(res['pearson_p']) < 0.05 else ''
                        p_spearman_class = 'text-danger fw-bold' if res['spearman_p'] != 'N/A' and not pd.isna(float(res['spearman_p'])) and float(res['spearman_p']) < 0.05 else ''

                        results_html += f"""
                            <tr>
                                <td>{res['b1']}</td>
                                <td>{res['b2']}</td>
                                <td>{res['n']}</td>
                                <td>{res['pearson_r']}</td>
                                <td class='{p_pearson_class}'>{res['pearson_p']}</td>
                                <td>{res['spearman_rho']}</td>
                                <td class='{p_spearman_class}'>{res['spearman_p']}</td>
                            </tr>
                        """
                        if res['error']:
                                results_html += f"<tr class='table-warning'><td colspan='7'><small>Note: {res['error']}</small></td></tr>"

                    results_html += "</tbody></table></div>"
                    results_html += "<p><small>Significant p-values (p < 0.05) are highlighted in bold red.</small></p>"

                return jsonify({'status': 'success', 'type': 'stats', 'data': results_html})


    except Exception as e:
        app.logger.error(f"Unhandled error in run_analysis for study {study_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'An internal server error occurred: {str(e)}'}), 500


@app.route('/finances/<study_id>', methods=['GET'])
@login_required
def finances(study_id):
    study = Study.query.get_or_404(study_id)
    if study.budget_amount is not None:
        budget_entry = FinancialLedger.query.filter_by(study_id=study_id, transaction_type='BUDGET').first()
        if not budget_entry:
            new_budget_entry = FinancialLedger(
                study_id=study_id,
                transaction_type='BUDGET',
                amount=study.budget_amount,
                transaction_date=study.start_date or datetime.now(timezone.utc).date(),
                description="Initial budget allocation."
            )
            db.session.add(new_budget_entry)
            db.session.commit()

    transactions = FinancialLedger.query.filter_by(study_id=study_id).order_by(FinancialLedger.transaction_date.asc()).all()
    expense_categories = ExpenseCategory.query.order_by(ExpenseCategory.category_name).all()
    total_budget = sum(t.amount for t in transactions if t.transaction_type in ['BUDGET', 'TOPUP'])
    total_expenses = sum(t.amount for t in transactions if t.transaction_type == 'EXPENSE')
    remaining_balance = total_budget - total_expenses
    today_date = datetime.now(timezone.utc).date()

    forecast_date = None
    if transactions and total_expenses > 0:
        first_transaction_date = min(t.transaction_date for t in transactions)
        days_elapsed = (today_date - first_transaction_date).days
        if days_elapsed > 0:
            avg_daily_spend = total_expenses / days_elapsed
            if avg_daily_spend > 0 and remaining_balance > 0:
                days_remaining = int(remaining_balance / avg_daily_spend)
                forecast_date = (today_date + timedelta(days=days_remaining)).strftime('%B %d, %Y')

    return render_template('finances.html', 
                           study=study, 
                           transactions=transactions,
                           total_budget=total_budget,
                           total_expenses=total_expenses,
                           remaining_balance=remaining_balance,
                           forecast_date=forecast_date,
                           expense_categories=expense_categories,
                           today_date=today_date)


@app.route('/finances/<study_id>/add_expense', methods=['POST'])
@login_required
def add_expense(study_id):
    _require_study_membership(study_id)
    Study.query.get_or_404(study_id)
    try:
        expense_date_str = request.form.get('expense_date')
        category_id = request.form.get('category_id')
        description = request.form.get('description')
        amount_str = request.form.get('amount')

        if not all([expense_date_str, category_id, description, amount_str]):
            flash('All fields are required to add an expense.', 'danger')
            return redirect(url_for('finances', study_id=study_id))

        new_expense = FinancialLedger(
            study_id=study_id,
            transaction_date=datetime.strptime(expense_date_str, '%Y-%m-%d').date(),
            transaction_type='EXPENSE',
            amount=Decimal(amount_str),
            description=description,
            category_id=category_id
        )
        db.session.add(new_expense)
        db.session.commit()
        flash('Expense recorded successfully.', 'success')
    except (ValueError, InvalidOperation) as e:
        db.session.rollback()
        flash(f'Invalid data provided. Please check your inputs. Error: {e}', 'danger')
    except Exception as e:
        db.session.rollback()
        flash(f'An unexpected error occurred: {e}', 'danger')
        
    return redirect(url_for('finances', study_id=study_id))


@app.route('/finances/<study_id>/topup_budget', methods=['POST'])
@login_required
def topup_budget(study_id):
    _require_study_membership(study_id)
    Study.query.get_or_404(study_id)
    try:
        topup_date_str = request.form.get('topup_date')
        description = request.form.get('description')
        amount_str = request.form.get('amount')

        if not all([topup_date_str, description, amount_str]):
            flash('All fields are required to top-up the budget.', 'danger')
            return redirect(url_for('finances', study_id=study_id))

        new_topup = FinancialLedger(
            study_id=study_id,
            transaction_date=datetime.strptime(topup_date_str, '%Y-%m-%d').date(),
            transaction_type='TOPUP',
            amount=Decimal(amount_str),
            description=description
        )
        db.session.add(new_topup)
        db.session.commit()
        flash('Budget topped up successfully.', 'success')
    except (ValueError, InvalidOperation) as e:
        db.session.rollback()
        flash(f'Invalid data provided. Please check your inputs. Error: {e}', 'danger')
    except Exception as e:
        db.session.rollback()
        flash(f'An unexpected error occurred: {e}', 'danger')
        
    return redirect(url_for('finances', study_id=study_id))


@app.route('/generate_demo_data', methods=['POST'])
@login_required
def generate_demo_data():
    g.disable_auditing = True # no audit during demo data creation
    
    blob_service_client = None
    eeg_generation_enabled = False
    connect_str = os.environ.get('AZURE_BLOB')
    if connect_str:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            app.logger.info("Demo Data: Azure Blob Service Client initialized.")
        except Exception as e:
            app.logger.error(f"Demo Data: Failed to initialize Azure Blob Client. {e}")
            flash('Error connecting to cloud storage. File-based demo data will be skipped.', 'warning')
    else:
        app.logger.warning("Demo Data: AZURE_BLOB env var not set. File-based demo data will be skipped.")
        flash('Cloud storage not configured. File-based demo data will be skipped.', 'warning')


    try:
        study_type_key = request.form.get('study_type')
        if not study_type_key or study_type_key not in DEMO_STUDY_DATA:
            flash('Invalid demo study type selected.', 'danger')
            return redirect(url_for('main_app'))

        if study_type_key in TRIAL_TYPE_TO_EEG_MAP and blob_service_client:
            eeg_generation_enabled = True
            app.logger.info(f"Demo Data: EEG Generation ENABLED for study type '{study_type_key}'")
        else:
            app.logger.info(f"Demo Data: EEG Generation DISABLED for study type '{study_type_key}' (Missing map or blob client)")


        selected_data = DEMO_STUDY_DATA[study_type_key]
        today = date.today()
        duration_days = int(selected_data['duration_months'] * 30.44) 
        end_date = today + timedelta(days=duration_days)

        biomarkers_from_db = BiomarkerType.query.all()
        biomarker_lookup = {b.biomarker_name: str(b.biomarker_id) for b in biomarkers_from_db}
        expense_categories_from_db = ExpenseCategory.query.all()
        category_lookup = {cat.category_name: cat.category_id for cat in expense_categories_from_db}
        app.logger.info(f"Demo Data: Found {len(category_lookup)} expense categories.")
        
        eeg_biomarker_name_to_id_map = {}
        if eeg_generation_enabled:
            eeg_biomarkers_from_db = BiomarkerType.query.filter(
                BiomarkerType.biomarker_name.in_(EEG_BIOMARKER_NAMES)
            ).all()
            eeg_biomarker_name_to_id_map = {b.biomarker_name: str(b.biomarker_id) for b in eeg_biomarkers_from_db}
            
            if len(eeg_biomarker_name_to_id_map) != len(EEG_BIOMARKER_NAMES):
                app.logger.warning("Demo Data: Mismatch between expected EEG biomarker names and DB.")
                app.logger.warning(f"Found: {list(eeg_biomarker_name_to_id_map.keys())}")


        biomarkers_to_gen_info = selected_data.get('biomarkers', [])
        biomarker_ids_to_allow = []
        for b_info in biomarkers_to_gen_info:
            b_id = biomarker_lookup.get(b_info['name'])
            if b_id:
                biomarker_ids_to_allow.append(b_id)
        
        biomarker_types_in_study = set(biomarker_ids_to_allow)
        if eeg_generation_enabled:
            for b_id in eeg_biomarker_name_to_id_map.values():
                biomarker_ids_to_allow.append(b_id)
                biomarker_types_in_study.add(b_id)
        
        allowed_biomarker_objects = BiomarkerType.query.filter(
            BiomarkerType.biomarker_id.in_(biomarker_ids_to_allow)
        ).all()
        
        budget_amount_demo = Decimal('50000.00') 

        new_study = Study(
            name=selected_data['name'],
            description=selected_data['description'],
            start_date=today,
            end_date=end_date,
            status='Planned',
            funding_source='Grant',
            budget_amount=budget_amount_demo,
            currency='USD',
            principal_investigator_id=str(current_user.id) 
        )
        db.session.add(new_study)
        db.session.flush()
        
        study_id_str = str(new_study.study_id)
        new_budget_entry = FinancialLedger(
            study_id=study_id_str,
            transaction_date=today,
            transaction_type='BUDGET',
            amount=budget_amount_demo,
            description='Initial demo budget'
        )
        db.session.add(new_budget_entry)

        new_settings = StudySettings(
            study_id=study_id_str,
            ai_enabled=False, 
            eeg_enabled=eeg_generation_enabled,
            wearables_enabled=False,
            biological_enabled=True,
            allowed_biomarkers=allowed_biomarker_objects
        )
        db.session.add(new_settings)

        financial_data = selected_data.get('financials', {})

        for topup in financial_data.get('topups', []):
            random_day_offset = random.randint(1, duration_days) if duration_days > 0 else 0
            trans_date = today + timedelta(days=random_day_offset)
            new_topup = FinancialLedger(
                study_id=study_id_str,
                transaction_date=trans_date,
                transaction_type='TOPUP',
                amount=Decimal(topup['amount']),
                description=topup['description']
            )
            db.session.add(new_topup)

        for expense in financial_data.get('expenses', []):
            category_id = category_lookup.get(expense['category'])
            if not category_id:
                app.logger.warning(f"Demo Data: Could not find category_id for '{expense['category']}'. Skipping expense.")
                continue
            
            random_day_offset = random.randint(1, duration_days) if duration_days > 0 else 0
            trans_date = today + timedelta(days=random_day_offset)
            new_expense = FinancialLedger(
                study_id=study_id_str,
                transaction_date=trans_date,
                transaction_type='EXPENSE',
                amount=Decimal(expense['amount']),
                description=expense['description'],
                category_id=category_id
            )
            db.session.add(new_expense)

        created_arms = []
        for arm_data in selected_data.get('arms', []):
            new_arm = StudyArm(
                study_id=study_id_str,
                arm_name=arm_data['arm_name'],
                description=arm_data['description']
            )
            db.session.add(new_arm)
            created_arms.append(new_arm)
        
        db.session.flush()

        arm_id_to_name_map = {str(c.arm_id): c.arm_name for c in created_arms}

        subject_counter = 1
        created_subjects = []
        for arm in created_arms:
            for _ in range(10): # 10 subjects per arm
                external_code = f"S{subject_counter:04d}"
                gender = random.choice(["Male", "Female"])
                
                if gender == "Male":
                    height = Decimal(random.uniform(165.0, 195.0)).quantize(Decimal('0.1'))
                    weight = Decimal(random.uniform(65.0, 110.0)).quantize(Decimal('0.1'))
                    preg_status = "Not Applicable"
                else:
                    height = Decimal(random.uniform(150.0, 180.0)).quantize(Decimal('0.1'))
                    weight = Decimal(random.uniform(50.0, 90.0)).quantize(Decimal('0.1'))
                    preg_status = "Not Pregnant"

                new_subject = Subject(
                    study_id=study_id_str,
                    arm_id=arm.arm_id,
                    external_subject_code=external_code,
                    first_name=random.choice(FAKE_FIRST_NAMES),
                    last_name=random.choice(FAKE_LAST_NAMES),
                    gender=gender,
                    height_cm=height,
                    weight_kg=weight,
                    consent_date=today,
                    status='Enrolled', # Start them as Enrolled
                    date_of_birth=date(random.randint(1950, 2000), random.randint(1, 12), random.randint(1, 28)),
                    ethnicity=random.choice(['Caucasian', 'Asian', 'Hispanic', 'African American', 'Other']),
                    race=random.choice(['White', 'Asian', 'Black or African American', 'Not Reported']),
                    handedness=random.choice(['Right', 'Left']),
                    pregnancy_status=preg_status,
                    education_level=random.choice(['Bachelors', 'Masters', 'High-School', 'PhD']),
                    employment_status=random.choice(['Full-Time', 'Part-Time', 'Student']),
                    marital_status=random.choice(['Single', 'Married', 'Partnered']),
                    smoking_status=random.choice(['Never', 'Quit', 'Low']),
                    alcohol_intake=random.choice(['Never', 'Low', 'Moderate']),
                    physical_activity_level=random.choice(['Low', 'Moderate', 'Sedentary'])
                )
                db.session.add(new_subject)
                created_subjects.append(new_subject)
                subject_counter += 1
        
        db.session.flush()

        biomarkers_to_gen = selected_data.get('biomarkers', [])
        symptoms_to_gen = selected_data.get('symptoms', [])
        aes_to_gen = selected_data.get('adverse_events', [])
        meds_to_gen = selected_data.get('medications', [])

        for subject in created_subjects:
            subject_id_str = str(subject.subject_id)

            def get_random_date():
                random_day_offset = random.randint(0, duration_days) if duration_days > 0 else 0
                return datetime.combine(today + timedelta(days=random_day_offset), datetime.min.time()) + timedelta(hours=random.randint(9, 17))

            for biomarker_info in biomarkers_to_gen:
                biomarker_name = biomarker_info['name']
                biomarker_id = biomarker_lookup.get(biomarker_name) 
                if not biomarker_id:
                    app.logger.warning(f"Demo Data: Could not find biomarker_id for '{biomarker_name}'. Skipping.")
                    continue
                
                for _ in range(random.randint(2, 4)): # 2-4 recordings per biomarker
                    value = Decimal(
                        random.uniform(biomarker_info['min'], biomarker_info['max'])
                    ).quantize(Decimal('0.01'))
                    
                    new_recording = StudyRecording(
                        study_id=study_id_str,
                        subject_id=subject_id_str,
                        recording_datetime=get_random_date(),
                        recording_type='Biomarker'
                    )
                    db.session.add(new_recording)
                    db.session.flush()
                    
                    new_biomarker_record = StudyRecordingBiomarker(
                        recording=new_recording,
                        study_id=study_id_str,
                        subject_id=subject_id_str,
                        biomarker_id=biomarker_id,
                        biomarker_value=value
                    )
                    db.session.add(new_biomarker_record)

            # Generate Symptom Recordings (for ~30% of subjects)
            if symptoms_to_gen and random.random() < 0.3:
                symptom = random.choice(symptoms_to_gen)
                new_recording = StudyRecording(
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    recording_datetime=get_random_date(),
                    recording_type='Symptom'
                )
                db.session.add(new_recording)
                db.session.flush()
                
                new_symptom = SubjectSymptom(
                    recording=new_recording,
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    symptom_verbatim=symptom['term'],
                    meddra_code='10019906', # Dummy code for "Headache"
                    meddra_term=symptom['term'],
                    severity=symptom['severity']
                )
                db.session.add(new_symptom)

            # Generate Adverse Event Recordings (for ~10% of subjects)
            if aes_to_gen and random.random() < 0.1:
                ae = random.choice(aes_to_gen)
                new_recording = StudyRecording(
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    recording_datetime=get_random_date(),
                    recording_type='Adverse Event'
                )
                db.session.add(new_recording)
                db.session.flush()
                
                new_ae = SubjectAdverseEvent(
                    recording=new_recording,
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    ae_verbatim=ae['term'],
                    meddra_code='10028813', # Dummy code for "Nausea"
                    meddra_term=ae['term'],
                    is_serious_ae=False,
                    severity_grade=ae['grade'],
                    causality='Possible',
                    outcome='Resolved'
                )
                db.session.add(new_ae)

            # Generate Medication Recordings (for ~20% of subjects)
            if meds_to_gen and random.random() < 0.2:
                med = random.choice(meds_to_gen)
                new_recording = StudyRecording(
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    recording_datetime=get_random_date(),
                    recording_type='Medication'
                )
                db.session.add(new_recording)
                db.session.flush()
                
                new_med = SubjectMedicationTaken(
                    recording=new_recording,
                    study_id=study_id_str,
                    subject_id=subject_id_str,
                    medication_name=med['name'],
                    dose=med['dose'],
                    route='Oral',
                    indication=med['indication'],
                    is_concomitant=True
                )
                db.session.add(new_med)

            # Generate plausible EEG file & derived biomarkers
            if eeg_generation_enabled:
                try:
                    arm_name = arm_id_to_name_map.get(str(subject.arm_id))
                    eeg_map = TRIAL_TYPE_TO_EEG_MAP.get(study_type_key, {'default': 'Relaxed'})
                    eeg_condition = eeg_map.get(arm_name, eeg_map['default'])

                    generate_demo_eeg_and_biomarkers(
                        subject=subject,
                        study_id_str=study_id_str,
                        eeg_condition=eeg_condition,
                        blob_service_client=blob_service_client,
                        biomarker_types_in_study=biomarker_types_in_study,
                        eeg_biomarker_map=eeg_biomarker_name_to_id_map
                    )
                except Exception as e:
                    app.logger.error(f"Demo EEG generation failed for subject {subject.external_subject_code}, rolling back transaction. Error: {e}")
                    raise e # Re-raise to trigger the main rollback


        participant = StudyParticipant.query.filter_by(email=current_user.email).first()
        participant_id_to_link = None
        if participant:
            participant_id_to_link = str(participant.participant_id)
            participant.role = 'Principal Investigator'
        else:
            new_participant = StudyParticipant(
                participant_id=str(current_user.id),
                email=current_user.email,
                first_name='Demo',
                last_name='Investigator',
                role='Principal Investigator'
            )
            db.session.add(new_participant)
            db.session.flush()
            participant_id_to_link = str(new_participant.participant_id)

        new_link = StudyParticipantLink(
            study_id=study_id_str,
            participant_id=participant_id_to_link,
            active=True,
            is_admin=True
        )
        db.session.add(new_link)
        
        db.session.commit()
        
        flash(f"Successfully generated demo study: '{selected_data['name']}'", 'success')

    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while generating demo data: {str(e)}', 'danger')
        app.logger.error(f"Error in generate_demo_data: {e}", exc_info=True)
        
    finally:
        g.disable_auditing = False # Always re-enable auditing

    return redirect(url_for('main_app'))


@app.route('/manage_account', methods=['GET', 'POST'])
@login_required
def manage_account():
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    if not participant:
        participant = StudyParticipant(
            participant_id=str(current_user.id),
            email=current_user.email,
            first_name="New",
            last_name="User"
        )
        db.session.add(participant)
        try:
            db.session.commit()
            flash('A new participant profile has been created for you. Please update your details.', 'info')
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating participant profile: {e}', 'danger')
            return redirect(url_for('main_app'))
            
    current_participant_id = str(participant.participant_id).lower()
    user_to_edit = participant
    user_account = current_user

    if request.method == 'POST':
        action = request.form.get('action')
        participant_id_to_edit = request.form.get('participant_id')
        if not participant_id_to_edit or participant_id_to_edit.lower() != current_participant_id:
            flash('You are not authorized to perform this action.', 'danger')
            return redirect(url_for('manage_account'))

        if action == 'update_details':
            try:
                new_email = request.form.get('email', '').strip().lower()
                if user_to_edit.email.lower() != new_email:
                    existing_user_check = User.query.filter(
                        User.email == new_email, 
                        User.id != user_account.id
                    ).first()
                    existing_participant_check = StudyParticipant.query.filter(
                        StudyParticipant.email == new_email,
                        StudyParticipant.participant_id != user_to_edit.participant_id
                    ).first()

                    if existing_user_check or existing_participant_check:
                        flash('That email address is already in use by another account.', 'danger')
                        return redirect(url_for('manage_account'))
                    
                    user_account.email = new_email
                    user_to_edit.email = new_email
                
                user_to_edit.role = request.form.get('role')
                user_to_edit.funding_role = request.form.get('funding_role')
                user_to_edit.percent_effort = request.form.get('percent_effort') or None
                user_to_edit.salary_contribution = request.form.get('salary_contribution') or None
                user_to_edit.first_name = request.form.get('first_name')
                user_to_edit.last_name = request.form.get('last_name')
                user_to_edit.phone = request.form.get('phone')
                user_to_edit.affiliation = request.form.get('affiliation')
                user_to_edit.department = request.form.get('department')
                user_to_edit.orcid = request.form.get('orcid')
                user_to_edit.country = request.form.get('country')
                
                # Handle AI API Key - save to User model
                ai_api_key = request.form.get('ai_api_key', '').strip()
                if ai_api_key:
                    user_account.ai_api_key = ai_api_key
                elif request.form.get('clear_api_key'):  # Optional: add checkbox to clear
                    user_account.ai_api_key = None

                db.session.commit()
                flash('Your details have been updated successfully.', 'success')

            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred: {e}', 'danger')
            
            return redirect(url_for('manage_account'))

        elif action == 'change_password':
            old_password = request.form.get('old_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not current_user.check_password(old_password):
                flash('Old password is incorrect.', 'danger')
                return redirect(url_for('manage_account'))
            
            if new_password != confirm_password:
                flash('New passwords do not match.', 'danger')
                return redirect(url_for('manage_account'))

            errors = []
            if len(new_password) < 8: errors.append("at least 8 characters")
            if not re.search(r"[A-Z]", new_password): errors.append("an uppercase letter")
            if not re.search(r"[a-z]", new_password): errors.append("a lowercase letter")
            if not re.search(r"\d", new_password): errors.append("a number")
            
            if errors:
                flash(f'Password must contain: {", ".join(errors)}.', 'danger')
                return redirect(url_for('manage_account'))

            current_user.set_password(new_password)
            db.session.commit()
            flash('Password changed successfully.', 'success')
            return redirect(url_for('manage_account'))

    user_to_edit_links = db.session.query(
        Study,
        StudyParticipantLink.is_admin
    ).join(
        StudyParticipantLink, Study.study_id == StudyParticipantLink.study_id
    ).filter(
        StudyParticipantLink.participant_id == current_participant_id
    ).order_by(Study.name).all()

    studies_data = []
    for study, is_admin_status in user_to_edit_links:
        studies_data.append({
            'study_id': study.study_id,
            'name': study.name,
            'is_admin': is_admin_status
        })

    is_admin = any(s['is_admin'] for s in studies_data)
    user_account_for_view = user_account
    
    roles_list = [
            "AI Specialist",
            "Biostatistician",
            "Clinical Data Analyst",
            "Clinical Research Associate",
            "Clinical Research Coordinator",
            "Clinical Trial Manager",
            "Co-Investigator",
            "Data Manager",
            "Data Programmer",
            "Data Safety Monitoring Board Member",
            "Data Scientist",
            "Enrollment Specialist",
            "Epidemiologist",
            "Ethics Committee Member",
            "Imaging Specialist",
            "Informed Consent Specialist",
            "Laboratory Technician",
            "Lead Investigator",
            "Medical Monitor",
            "Medical Writer",
            "Pharmacist",
            "Pharmacokineticist",
            "Pharmacologist",
            "Principal Investigator",
            "Project Manager",
            "Protocol Developer",
            "Publication Coordinator",
            "Quality Assurance Specialist",
            "Regulatory Affairs Specialist",
            "Regulatory Monitor",
            "Research Assistant",
            "Scientific Advisor",
            "Sponsor",
            "Study Nurse",
            "Sub-Investigator"
        ]
    
    return render_template(
        'manage_account.html',
        is_admin=is_admin,
        user_to_edit=user_to_edit,
        user_account_to_edit=user_account_for_view,
        studies_data=studies_data,
        current_participant_id=current_participant_id,
        current_admin_details=participant,
        roles_list=roles_list,
        all_participants=[]
    )


@app.route('/audit_log', methods=['GET', 'POST'])
@login_required
def view_audit_log():
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    is_admin = False
    if participant:
        admin_link_exists = StudyParticipantLink.query.filter_by(
            participant_id=participant.participant_id,
            is_admin=True
        ).first()
        if admin_link_exists:
            is_admin = True
    
    if not is_admin:
        flash('You are not authorized to access this page.', 'danger')
        return redirect(url_for('main_app'))

    if request.method == 'POST':
        if request.form.get('action') == 'clear_log':
            try:
                db.session.query(AuditLog).delete()
                db.session.commit()
                flash('Audit log has been successfully cleared.', 'success')
            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred while clearing the log: {e}', 'danger')
            return redirect(url_for('view_audit_log'))

    logs_data = db.session.query(
        AuditLog,
        Study.name,
        Subject.external_subject_code
    ).outerjoin(
        Study, AuditLog.study_id == Study.study_id
    ).outerjoin(
        Subject, AuditLog.subject_id == Subject.subject_id
    ).order_by(
        AuditLog.change_datetime.desc()
    ).all()
    return render_template('view_audit_log.html', logs_data=logs_data)


@app.route('/manage_account/update_study_perms', methods=['POST'])
@login_required
def update_study_perms():
    """
    Handles saving the admin permissions from the manage_account page.
    """
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    participant_id_to_edit = request.form.get('participant_id')
    
    if not participant or not participant_id_to_edit:
        flash('Invalid request. User not found.', 'danger')
        return redirect(url_for('main_app'))
        
    current_participant_id = str(participant.participant_id).lower()
    admin_links = db.session.query(StudyParticipantLink.study_id).filter(
        StudyParticipantLink.participant_id == current_participant_id,
        StudyParticipantLink.is_admin == True
    ).all()
    admin_rights_for_studies = {link.study_id for link in admin_links}

    links_to_edit = StudyParticipantLink.query.filter_by(
        participant_id=participant_id_to_edit.lower()
    ).all()

    checked_admin_studies = set(request.form.getlist('is_admin_for_study'))
    
    updated_count = 0
    for link in links_to_edit:
        if link.study_id in admin_rights_for_studies:
            is_now_admin = link.study_id in checked_admin_studies
            if link.is_admin != is_now_admin:
                link.is_admin = is_now_admin
                updated_count += 1

    try:
        db.session.commit()
        if updated_count > 0:
            flash(f'Successfully updated admin permissions for {updated_count} studies.', 'success')
        else:
            flash('No admin permission changes were made.', 'info')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while updating permissions: {e}', 'danger')

    return redirect(url_for('manage_account', selected_user_id=participant_id_to_edit))


def _resolve_audit_json(data_dict):
    """Helper function to resolve foreign keys within audit JSON."""
    if not data_dict:
        return data_dict
    
    resolved_dict = data_dict.copy()
    
    try:
        if 'biomarker_id' in resolved_dict:
            b_id = resolved_dict['biomarker_id']
            biomarker = BiomarkerType.query.get(b_id)
            resolved_dict['biomarker_name'] = f"{biomarker.biomarker_name} (ID: {b_id})" if biomarker else f"Unknown (ID: {b_id})"

        if 'eeg_id' in resolved_dict:
            e_id = resolved_dict['eeg_id']
            eeg = EEG.query.get(e_id)
            resolved_dict['eeg_device'] = f"{eeg.manufacturer} {eeg.device_type} (ID: {e_id})" if eeg else f"Unknown (ID: {e_id})"

        if 'wearable_id' in resolved_dict:
            w_id = resolved_dict['wearable_id']
            wearable = Wearable.query.get(w_id)
            resolved_dict['wearable_device'] = f"{wearable.manufacturer} {wearable.device_name} (ID: {w_id})" if wearable else f"Unknown (ID: {w_id})"
        if 'image_type' in resolved_dict:
             resolved_dict['image_details'] = f"Image Type: {resolved_dict['image_type']}"
    except Exception:
        pass  # unique identifiers still remaining from objects probably removed earlier, so just don't resolve these
  
    return resolved_dict


@app.route('/api/audit_log/<audit_log_id>', methods=['GET'])
@login_required
def get_audit_log_details(audit_log_id):
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()

    is_admin = False
    if participant:
        admin_link_exists = StudyParticipantLink.query.filter_by(
            participant_id=participant.participant_id,
            is_admin=True
        ).first()
        if admin_link_exists:
            is_admin = True
    
    if not is_admin:
        return jsonify(error="Not authorized"), 403
    
    log = AuditLog.query.get_or_404(audit_log_id)
    old_data = json.loads(log.old_value) if log.old_value else {}
    new_data = json.loads(log.new_value) if log.new_value else {}
    resolved_old = _resolve_audit_json(old_data)
    resolved_new = _resolve_audit_json(new_data)
    return jsonify(old_value=resolved_old, new_value=resolved_new)


def _delete_study_and_data(study_id_to_delete):
    """
    Internal helper function to delete a study and all its related data,
    including associated Azure Blobs.
    Assumes the study object exists. USE WITH CAUTION.
    Returns True on success, False on failure.
    """
    study = Study.query.get(study_id_to_delete)
    if not study:
        app.logger.error(f"Attempted to delete non-existent study ID: {study_id_to_delete}")
        return False

    try:
        eeg_uris = db.session.query(StudyRecordingEEG.data_uri)\
            .filter(StudyRecordingEEG.study_id == study_id_to_delete).all()
        wearable_uris = db.session.query(StudyRecordingWearable.data_uri)\
            .filter(StudyRecordingWearable.study_id == study_id_to_delete).all()
        image_uris = db.session.query(StudyRecordingImage.data_uri)\
            .filter(StudyRecordingImage.study_id == study_id_to_delete).all()

        all_uris = [uri[0] for uri in eeg_uris + wearable_uris + image_uris if uri[0]]

        connect_str = os.environ.get('AZURE_BLOB')
        if all_uris and connect_str:
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
                container_name = "recordings"
                for data_uri in all_uris:
                    try:
                        blob_name = '/'.join(data_uri.split('/')[4:])
                        if blob_name:
                            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                            blob_client.delete_blob(delete_snapshots="include")
                            app.logger.info(f"Deleted blob: {blob_name} for study {study_id_to_delete}")
                    except ResourceNotFoundError:
                        app.logger.warning(f"Blob not found, skipping deletion: {blob_name} for study {study_id_to_delete}")
                    except Exception as blob_ex:
                        app.logger.error(f"Failed to delete blob {blob_name} for study {study_id_to_delete}: {blob_ex}")
            except Exception as service_ex:
                app.logger.error(f"Failed to connect to Azure Blob Service for study {study_id_to_delete} cleanup: {service_ex}")
        elif all_uris and not connect_str:
            app.logger.warning(f"Recordings found for study {study_id_to_delete}, but Azure connection string not configured. Blobs will not be deleted.")

        subjects = Subject.query.with_entities(Subject.subject_id).filter_by(study_id=study_id_to_delete).all()
        subject_ids = [s[0] for s in subjects]

        if subject_ids:
            StudyRecordingEEG.query.filter(StudyRecordingEEG.study_id == study_id_to_delete).delete(synchronize_session=False)
            StudyRecordingWearable.query.filter(StudyRecordingWearable.study_id == study_id_to_delete).delete(synchronize_session=False)
            StudyRecordingImage.query.filter(StudyRecordingImage.study_id == study_id_to_delete).delete(synchronize_session=False)
            StudyRecordingBiomarker.query.filter(StudyRecordingBiomarker.study_id == study_id_to_delete).delete(synchronize_session=False)
            StudyRecording.query.filter(StudyRecording.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectClinician.query.filter(SubjectClinician.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectConsent.query.filter(SubjectConsent.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectContact.query.filter(SubjectContact.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectDiagnosis.query.filter(SubjectDiagnosis.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectMedication.query.filter(SubjectMedication.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            SubjectDocument.query.filter(SubjectDocument.subject_id.in_(subject_ids)).delete(synchronize_session=False)
            Subject.query.filter(Subject.subject_id.in_(subject_ids)).delete(synchronize_session=False)

        StudyDocument.query.filter_by(study_id=study_id_to_delete).delete(synchronize_session=False)
        FinancialLedger.query.filter_by(study_id=study_id_to_delete).delete(synchronize_session=False)
        StudyArm.query.filter_by(study_id=study_id_to_delete).delete(synchronize_session=False)
        StudyParticipantLink.query.filter_by(study_id=study_id_to_delete).delete(synchronize_session=False)
        StudySettings.query.filter_by(study_id=study_id_to_delete).delete(synchronize_session=False)
        db.session.delete(study)
        app.logger.info(f"Successfully queued database deletion for study ID: {study_id_to_delete}")
        return True

    except Exception as e:
        app.logger.error(f"Error during database deletion in _delete_study_and_data for study {study_id_to_delete}: {e}")
        return False


@app.route('/close_account/<user_id_to_delete>', methods=['POST'])
@login_required
def close_account(user_id_to_delete):
    """
    Handles the permanent deletion of a user's own account and associated participant data.
    If deleting the user leaves a study with no participants OR no admins, the action is blocked.
    """
    user_account = User.query.get_or_404(user_id_to_delete)
    
    # Check that the user being deleted is the one who is logged in.
    if str(current_user.id) != user_id_to_delete:
        flash('You are not authorized to perform this action.', 'danger')
        return redirect(url_for('manage_account'))

    participant_to_delete = StudyParticipant.query.filter_by(email=user_account.email).first()

    if not participant_to_delete:
        # This is a fallback in case a user exists but has no participant profile
        flash('Participant profile not found. Deleting login only.', 'warning')
        try:
            db.session.delete(user_account)
            db.session.commit()
            logout_user()
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting orphaned user account: {e}', 'danger')
            return redirect(url_for('manage_account'))

    participant_id_str = str(participant_to_delete.participant_id).lower()
    studies_to_potentially_delete = []
    user_links = StudyParticipantLink.query.filter_by(participant_id=participant_id_str).all()
    
    # Check if this user is the last admin for any study they are part of.
    studies_where_last_admin = []
    admin_links = StudyParticipantLink.query.filter_by(participant_id=participant_id_str, is_admin=True).all()

    for admin_link in admin_links:
        study_id = admin_link.study_id
        # Check for *other* admins in the same study
        other_admin_exists = StudyParticipantLink.query.filter(
            StudyParticipantLink.study_id == study_id,
            StudyParticipantLink.is_admin == True,
            StudyParticipantLink.participant_id != participant_id_str
        ).first()

        if not other_admin_exists:
            # This user is the last admin for this study
            study = Study.query.get(study_id)
            studies_where_last_admin.append(study.name if study else "an unknown study")

    if studies_where_last_admin:
        studies_list = ", ".join([f'"{name}"' for name in studies_where_last_admin])
        flash(f'Cannot close account. You are the last remaining administrator for the following study/studies: {studies_list}. Please promote another user to admin in each study before closing your account.', 'danger')
        return redirect(url_for('manage_account'))

    linked_study_ids = {link.study_id for link in user_links}
    for study_id in linked_study_ids:
        other_participants_count = StudyParticipantLink.query.filter(
            StudyParticipantLink.study_id == study_id,
            StudyParticipantLink.participant_id != participant_id_str
        ).count()

        if other_participants_count == 0:
            studies_to_potentially_delete.append(study_id)

    try:
        deleted_study_names = []
        for study_id in studies_to_potentially_delete:
            study = Study.query.get(study_id)
            if study:
                study_name = study.name
                success = _delete_study_and_data(study_id)
                if success:
                    deleted_study_names.append(study_name)
                else:
                    raise RuntimeError(f"Failed to delete orphaned study {study_id}")

        StudyParticipantLink.query.filter_by(participant_id=participant_id_str).delete(synchronize_session=False)
        db.session.delete(participant_to_delete)
        db.session.delete(user_account)
        db.session.commit()

        success_message = 'Your account has been closed successfully.'
        if deleted_study_names:
            studies_list_str = ", ".join([f'"{name}"' for name in deleted_study_names])
            success_message += f' The following studies were also deleted as you were the last participant: {studies_list_str}.'
        
        logout_user()
        flash(success_message, 'success')
        return redirect(url_for('index'))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error during close_account transaction for user {user_id_to_delete}: {e}")
        flash(f'An error occurred while closing the account: {e}', 'danger')
        return redirect(url_for('manage_account'))


def calculate_financial_summary(study_id):
    """Calculates financial totals for a study."""
    transactions = FinancialLedger.query.filter_by(study_id=study_id).all()
    total_budget = sum(t.amount for t in transactions if t.transaction_type in ['BUDGET', 'TOPUP'])
    total_expenses = sum(t.amount for t in transactions if t.transaction_type == 'EXPENSE')
    remaining_balance = total_budget - total_expenses
    forecast_date_str = None
    today_date = datetime.now(timezone.utc).date()
    if transactions and total_expenses > 0:
        first_transaction_date = min(t.transaction_date for t in transactions)
        days_elapsed = (today_date - first_transaction_date).days
        if days_elapsed > 0:
            avg_daily_spend = total_expenses / days_elapsed
            if avg_daily_spend > 0 and remaining_balance > 0:
                days_remaining = int(remaining_balance / avg_daily_spend)
                forecast_date = (today_date + timedelta(days=days_remaining))
                forecast_date_str = forecast_date.strftime('%Y-%m-%d')

    return {
        'total_budget': total_budget,
        'total_expenses': total_expenses,
        'remaining_balance': remaining_balance,
        'forecast_date': forecast_date_str,
        'transactions': transactions
    }


@app.route('/study_dashboard/<study_id>', methods=['GET'])
@login_required
def study_dashboard(study_id):
    study = Study.query.get_or_404(study_id)

    total_subjects = Subject.query.filter_by(study_id=study_id).count()
    active_subjects = Subject.query.filter_by(study_id=study_id, withdrawal_date=None).count()
    total_recordings = StudyRecording.query.filter_by(study_id=study_id).count()
    financials = calculate_financial_summary(study_id)
    kpis = {
        'status': study.status,
        'total_subjects': total_subjects,
        'active_subjects': active_subjects,
        'start_date': study.start_date.strftime('%Y-%m-%d') if study.start_date else 'N/A',
        'end_date': study.end_date.strftime('%Y-%m-%d') if study.end_date else 'N/A',
        'total_recordings': total_recordings,
        'remaining_balance': financials['remaining_balance']
    }

    plots_to_embed = {}
    
    BOKEH_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    burn_rate_df = pd.DataFrame([{
        'date': t.transaction_date,
        'expense': float(t.amount) if t.transaction_type == 'EXPENSE' else 0.0,
        'budget': float(t.amount) if t.transaction_type in ['BUDGET', 'TOPUP'] else 0.0
        } for t in financials['transactions']])

    if not burn_rate_df.empty:
        burn_rate_df['date'] = pd.to_datetime(burn_rate_df['date'])
        burn_rate_df = burn_rate_df.sort_values('date')
        burn_rate_df['Cumulative Expense'] = burn_rate_df['expense'].cumsum()
        burn_rate_df['Cumulative Budget'] = burn_rate_df['budget'].cumsum()
        
        try:
            source = ColumnDataSource(burn_rate_df)
            p_burn = figure(
                title='Financial Burn Rate vs Budget',
                x_axis_type='datetime',
                height=350,
                sizing_mode='stretch_width',
                tools=BOKEH_TOOLS
            )
            
            p_burn.line(x='date', y='Cumulative Expense', source=source, legend_label='Cumulative Expense', color='#20519c', line_width=2)
            p_burn.line(x='date', y='Cumulative Budget', source=source, legend_label='Cumulative Budget', color='#4caf50', line_dash='dashed', line_width=2)
            
            budget_line = Span(location=float(financials['total_budget']), dimension='width',
                               line_color='red', line_dash='dotted', line_width=2)
            p_burn.add_layout(budget_line)
            p_burn.line(x=[], y=[], legend_label="Total Budget", line_color="red", line_dash="dotted")

            p_burn.xaxis.axis_label = 'Date'
            p_burn.yaxis.axis_label = 'Amount ($)'
            p_burn.yaxis.formatter = NumeralTickFormatter(format="$,.00")
            p_burn.legend.location = 'top_left'
            p_burn.legend.click_policy = 'hide'
            
            p_burn.add_tools(HoverTool(
                tooltips=[
                    ('Date', '@date{%F}'),
                    ('Expense', '$@{%Cumulative Expense}{%0.2f}'),
                    ('Budget', '$@{%Cumulative Budget}{%0.2f}'),
                ],
                formatters={'@date': 'datetime', '@{%Cumulative Expense}': 'printf', '@{%Cumulative Budget}': 'printf'},
                mode='vline'
            ))
            plots_to_embed['burn_rate_chart'] = p_burn 

        except Exception as e:
            app.logger.error(f"Error creating burn rate Bokeh chart: {e}", exc_info=True)
    else:
        app.logger.warning("Burn rate DataFrame was empty. Skipping chart generation.")

    enrollment_query = db.session.query(Subject.consent_date)\
        .filter(Subject.study_id == study_id, Subject.consent_date != None)\
        .order_by(Subject.consent_date)
    enrollment_dates = [r[0] for r in enrollment_query.all() if r[0] is not None]

    if enrollment_dates:
        enrollment_df = pd.DataFrame({'consent_date': enrollment_dates})
        enrollment_df['consent_date'] = pd.to_datetime(enrollment_df['consent_date'])
        enrollment_df = enrollment_df.sort_values('consent_date')
        enrollment_df['Cumulative Enrollment'] = range(1, len(enrollment_df) + 1)

        try:
            source = ColumnDataSource(enrollment_df)
            p_enroll = figure(
                title='Cumulative Subject Enrollment',
                x_axis_type='datetime',
                height=350,
                sizing_mode='stretch_width',
                tools=BOKEH_TOOLS
            )
            
            p_enroll.line(x='consent_date', y='Cumulative Enrollment', source=source, line_width=2, color='#20519c')
            p_enroll.scatter(x='consent_date', y='Cumulative Enrollment', source=source, size=4, color='#20519c', fill_alpha=0.6)
            
            p_enroll.xaxis.axis_label = 'Date'
            p_enroll.yaxis.axis_label = 'Number of Subjects'
            
            p_enroll.add_tools(HoverTool(
                tooltips=[
                    ('Date', '@consent_date{%F}'),
                    ('Enrolled', '@{%Cumulative Enrollment}'),
                ],
                formatters={'@consent_date': 'datetime'},
                mode='vline'
            ))
            
            plots_to_embed['enrollment_chart'] = p_enroll
            app.logger.info("Successfully generated enrollment Bokeh chart.")
        except Exception as e:
            app.logger.error(f"Error creating enrollment Bokeh chart: {e}", exc_info=True)
    else:
        app.logger.warning("No enrollment dates found. Skipping enrollment chart generation.")

    arm_dist_query = db.session.query(StudyArm.arm_name, func.count(Subject.subject_id).label('count'))\
        .outerjoin(Subject, StudyArm.arm_id == Subject.arm_id)\
        .filter(StudyArm.study_id == study_id)\
        .group_by(StudyArm.arm_id, StudyArm.arm_name)
    arm_dist_df = pd.read_sql(arm_dist_query.statement, db.engine)

    if not arm_dist_df.empty and arm_dist_df['count'].sum() > 0:
        arm_dist_df['arm_name'] = arm_dist_df['arm_name'].fillna('Unassigned')
        pie_data = arm_dist_df.groupby('arm_name')['count'].sum().reset_index(name='count')
        
        try:
            pie_data['angle'] = pie_data['count'] / pie_data['count'].sum() * 2 * math.pi
            pie_data['percentage'] = (pie_data['count'] / pie_data['count'].sum() * 100)
            
            num_colors = len(pie_data)
            if num_colors <= 20:
                palette_key = max(3, num_colors)
                colors = Category20c[palette_key][:num_colors]
            else:
                colors = Viridis256[:num_colors]
            pie_data['color'] = colors

            source = ColumnDataSource(pie_data)
            p_pie = figure(
                title="Subject Distribution by Arm",
                height=350,
                tools=BOKEH_TOOLS,
                toolbar_location="right",
                tooltips="@arm_name: @count subjects (@percentage{0.1f}%)",
                x_range=(-0.5, 1.5) 
            )

            p_pie.annular_wedge(
                x=0, y=0, 
                inner_radius=0.25,
                outer_radius=0.4,
                start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'),
                line_color="white",
                fill_color='color',
                legend_field='arm_name',
                source=source
            )
            
            p_pie.axis.axis_label = None
            p_pie.axis.visible = False
            p_pie.grid.grid_line_color = None
            p_pie.legend.location = "center_right"
            
            plots_to_embed['arm_pie_chart'] = p_pie
            app.logger.info("Successfully generated arm donut Bokeh chart.")
        except Exception as e:
            app.logger.error(f"Error creating arm donut Bokeh chart: {e}", exc_info=True)
    elif arm_dist_df.empty:
         app.logger.warning("Arm distribution DataFrame was empty. Skipping chart generation.")
    else:
         app.logger.warning("Arm distribution DataFrame has zero total subjects. Skipping chart generation.")

    recordings_sched_query = db.session.query(
        Subject.external_subject_code,
        StudyRecording.recording_datetime,
    ).join(Subject, StudyRecording.subject_id == Subject.subject_id)\
     .filter(StudyRecording.study_id == study_id)\
     .order_by(Subject.external_subject_code, StudyRecording.recording_datetime)
    recordings_sched_df = pd.read_sql(recordings_sched_query.statement, db.engine)
    
    if not recordings_sched_df.empty:
        recordings_sched_df['recording_datetime'] = pd.to_datetime(recordings_sched_df['recording_datetime'])
        recordings_sched_df['Week'] = recordings_sched_df['recording_datetime'].dt.strftime('%Y-W%U')
        heatmap_data = recordings_sched_df.groupby(['external_subject_code', 'Week']).size().unstack(fill_value=0)
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
        if len(heatmap_data.columns) > 8:
            heatmap_data = heatmap_data.iloc[:, -8:]

        try:
            heatmap_df = heatmap_data.stack().reset_index(name='count')
            source = ColumnDataSource(heatmap_df)
            
            subjects_list = heatmap_data.index.tolist()
            weeks_list = heatmap_data.columns.tolist()
            
            plot_height = max(250, len(subjects_list) * 20)
            plot_width = max(600, len(weeks_list) * 30)
            
            mapper = LinearColorMapper(palette=Viridis256, low=0, high=heatmap_df['count'].max())

            p_heat = figure(
                title='Recording Activity Heatmap (Recordings per Week)',
                x_range=FactorRange(factors=weeks_list),
                y_range=FactorRange(factors=subjects_list),
                height=plot_height,
                width=plot_width,
                tools=BOKEH_TOOLS,
                toolbar_location="right"
            )

            p_heat.rect(
                x='Week', y='external_subject_code', width=1, height=1, source=source,
                fill_color={'field': 'count', 'transform': mapper},
                line_color='lightgray'
            )
            
            color_bar = ColorBar(color_mapper=mapper, label_standoff=12,
                                 border_line_color=None, location=(0, 0),
                                 title="Recordings")
            p_heat.add_layout(color_bar, 'right')
            
            p_heat.add_tools(HoverTool(
                tooltips=[
                    ('Subject', '@external_subject_code'),
                    ('Week', '@Week'),
                    ('Recordings', '@count')
                ]
            ))

            p_heat.xaxis.axis_label = 'Week of Study'
            p_heat.yaxis.axis_label = 'Subject Code'
            p_heat.xaxis.major_label_orientation = math.pi / 3
            p_heat.yaxis.major_label_orientation = "horizontal"
            
            plots_to_embed['recording_heatmap'] = p_heat
            app.logger.info("Successfully generated recording heatmap Bokeh chart.")
        except Exception as e:
            app.logger.error(f"Error creating recording heatmap: {e}", exc_info=True)
    else:
        app.logger.warning("Recording schedule DataFrame was empty. Skipping heatmap generation.")

    flagged_subjects = []
    inactivity_threshold = datetime.now(timezone.utc) - timedelta(days=30)
    withdrawn = Subject.query.filter(Subject.study_id == study_id, Subject.withdrawal_date != None).all()
    for s in withdrawn:
        flagged_subjects.append({
            'code': s.external_subject_code,
            'status': 'Withdrawn',
            'last_activity': s.withdrawal_date.strftime('%Y-%m-%d') if s.withdrawal_date else 'N/A'
        })
    subq = db.session.query(StudyRecording.subject_id, func.max(StudyRecording.recording_datetime).label('last_rec_date'))\
        .filter(StudyRecording.study_id == study_id).group_by(StudyRecording.subject_id).subquery()
    inactive = db.session.query(Subject, subq.c.last_rec_date)\
        .outerjoin(subq, Subject.subject_id == subq.c.subject_id)\
        .filter(
            Subject.study_id == study_id,
            Subject.withdrawal_date == None,
            (subq.c.last_rec_date == None) | (subq.c.last_rec_date < inactivity_threshold)
        ).order_by(Subject.external_subject_code).all()
    for s, last_date in inactive:
        if not any(f['code'] == s.external_subject_code for f in flagged_subjects):
             flagged_subjects.append({
                'code': s.external_subject_code,
                'status': 'Inactive (No recording in >30 days)',
                'last_activity': last_date.strftime('%Y-%m-%d %H:%M') if last_date else 'Never'
            })
    flagged_subjects.sort(key=lambda x: x['code'])
    Subj = aliased(Subject)

    recent_aes_query = db.session.query(
        SubjectAdverseEvent.meddra_term,
        SubjectAdverseEvent.severity_grade,
        StudyRecording.recording_datetime,
        Subj.external_subject_code
    ).join(
        StudyRecording, SubjectAdverseEvent.recording_id == StudyRecording.recording_id
    ).join(
        Subj, StudyRecording.subject_id == Subj.subject_id
    ).filter(
        StudyRecording.study_id == study_id
    ).order_by(
        desc(StudyRecording.recording_datetime)
    ).limit(5).all()

    recent_aes = [
        {'term': r.meddra_term, 'grade': r.severity_grade, 'date': r.recording_datetime.strftime('%Y-%m-%d'), 'code': r.external_subject_code}
        for r in recent_aes_query
    ]

    recent_symptoms_query = db.session.query(
        SubjectSymptom.meddra_term,
        SubjectSymptom.severity,
        StudyRecording.recording_datetime,
        Subj.external_subject_code
    ).join(
        StudyRecording, SubjectSymptom.recording_id == StudyRecording.recording_id
    ).join(
        Subj, StudyRecording.subject_id == Subj.subject_id
    ).filter(
        StudyRecording.study_id == study_id
    ).order_by(
        desc(StudyRecording.recording_datetime)
    ).limit(5).all()

    recent_symptoms = [
        {'term': r.meddra_term, 'severity': r.severity, 'date': r.recording_datetime.strftime('%Y-%m-%d'), 'code': r.external_subject_code}
        for r in recent_symptoms_query
    ]

    study_settings = StudySettings.query.get(study_id)
    allowed_biomarker_ids_for_chart = []
    if study_settings and study_settings.biological_enabled:
        allowed_biomarkers_in_settings = study_settings.allowed_biomarkers
        biological_biomarkers_allowed = [b for b in allowed_biomarkers_in_settings if b.sample_type.lower() not in ['eeg', 'wearable']]
        allowed_biomarker_ids_for_chart = [b.biomarker_id for b in biological_biomarkers_allowed]

    if allowed_biomarker_ids_for_chart:
        biomarker_freq_query = db.session.query(
            BiomarkerType.biomarker_name,
            func.count(StudyRecordingBiomarker.recording_id).label('count')
        ).select_from(StudyRecordingBiomarker)\
         .join(BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id)\
         .filter(
             StudyRecordingBiomarker.study_id == study_id
         )\
         .group_by(BiomarkerType.biomarker_name)\
         .order_by(func.count().desc())
        biomarker_freq_df = pd.read_sql(biomarker_freq_query.statement, db.engine)
        
        if not biomarker_freq_df.empty:
            try:
                biomarker_freq_df = biomarker_freq_df.sort_values('count', ascending=True)
                source = ColumnDataSource(biomarker_freq_df)
                factors = biomarker_freq_df['biomarker_name'].tolist()

                p_bio = figure(
                    title='Biomarker Recordings Total',
                    y_range=FactorRange(factors=factors),
                    height=max(250, len(factors) * 30), 
                    sizing_mode='stretch_width',
                    tools=BOKEH_TOOLS
                )
                
                p_bio.hbar(y='biomarker_name', right='count', source=source, height=0.8, color="#20519c")
                
                p_bio.xaxis.axis_label = 'Number of Recordings'
                p_bio.yaxis.axis_label = 'Biomarker'
                
                labels = LabelSet(x='count', y='biomarker_name', text='count',
                                  x_offset=5, y_offset=-7, 
                                  source=source, text_font_size="9pt", text_color="#555555")
                p_bio.add_layout(labels)

                p_bio.add_tools(HoverTool(
                    tooltips=[
                        ('Biomarker', '@biomarker_name'),
                        ('Recordings', '@count')
                    ],
                    mode='hline'
                ))
                
                plots_to_embed['biomarker_freq_chart'] = p_bio
                app.logger.info("Successfully generated biomarker frequency Bokeh chart.")
            except Exception as e:
                app.logger.error(f"Error creating biomarker frequency Bokeh chart: {e}", exc_info=True)
        else:
             app.logger.warning("Biomarker frequency DataFrame was empty after query. Skipping chart generation.")
    else:
         app.logger.warning("No allowed biological biomarker IDs found for frequency chart. Skipping chart generation.")

    plot_script = None
    plot_divs = {}
    try:
        if plots_to_embed:
            plot_script, plot_divs = components(plots_to_embed)
            app.logger.info(f"Successfully generated components for: {list(plots_to_embed.keys())}")
        else:
            app.logger.warning("No plots were generated to embed.")
    except Exception as e:
        app.logger.error(f"Error calling Bokeh components(): {e}", exc_info=True)
    
    return render_template(
        'study_dashboard.html',
        study=study,
        kpis=kpis,
        plot_script=plot_script,
        plot_divs=plot_divs,
        flagged_subjects=flagged_subjects,
        recent_aes=recent_aes,
        recent_symptoms=recent_symptoms
    )


def _download_blob_to_zip(zf, data_uri, zip_path, blob_service_client):
    if not blob_service_client or not data_uri:
        app.logger.warning(f"Skipping blob write for {zip_path}: Blob service or URI is missing.")
        return

    try:
        # Assumes URI is like: https://[account].blob.core.windows.net/[container]/[blob_path]
        container_name = data_uri.split('/')[3]
        blob_name = '/'.join(data_uri.split('/')[4:])
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        downloader = blob_client.download_blob()
        blob_data = downloader.readall()
        zf.writestr(zip_path, blob_data)
        
    except ResourceNotFoundError:
        app.logger.error(f"Blob not found in Azure, skipping: {data_uri}")
        zf.writestr(zip_path + ".ERROR_NOT_FOUND.txt", f"The file at {data_uri} was not found in Azure Storage.")
    except Exception as e:
        app.logger.error(f"Failed to download or write blob {data_uri} to zip: {e}")
        zf.writestr(zip_path + f".ERROR_{e}.txt", f"Failed to download {data_uri}.")


@app.route('/study/<study_id>/download_multimodal_packet', methods=['POST'])
@login_required
def download_multimodal_packet(study_id):
    """
    Exports a multimodal analysis packet as a ZIP containing:
    - A wide-format biomarkers CSV with one row per subject per timepoint
    - Raw EEG, Wearable, and Imaging files organised by subject/date
    
    CSV Structure (matches the canonical import/export template):
        Arm, Subject, DateTime, {Biomarker columns...}
    
    Biomarker column naming:
        - If sample_type exists: "{SampleType}_{BiomarkerName}" → e.g. Blood_CRP, EEG_AlphaPower, HRV_RMSSD
        - If sample_type is NULL/empty: "{BiomarkerName}" alone → e.g. Ferritin, GAD7, IL6
        
    This preserves the exact naming convention from the biomarker_types table,
    where some biomarkers have an explicit sample_type prefix and others don't.
    """
    try:
        study = Study.query.get_or_404(study_id)
        subject_ids = request.form.getlist('subject_ids')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')

        if not subject_ids:
            flash('You must select at least one subject to export.', 'danger')
            return redirect(url_for('study_recordings', study_id=study_id))
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else date.min
        except ValueError:
            start_date = date.min
        try:
            end_date = (datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)).date() if end_date_str else date.max
        except ValueError:
            end_date = date.max

        # --- Azure Blob Storage connection ---
        connect_str = os.environ.get('AZURE_BLOB')
        blob_service_client = None
        if connect_str:
            try:
                blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            except Exception as e:
                app.logger.error(f"Failed to connect to Azure Blob Storage: {e}")
                flash(f"Error connecting to cloud storage: {e}. File downloads will be skipped.", 'warning')
        else:
            app.logger.warning("AZURE_BLOB environment variable not set. File downloads will be skipped.")
            flash("Cloud storage is not configured. Raw files will be skipped.", 'warning')

        # =================================================================
        # STEP 1: Build the wide-format biomarker CSV
        # =================================================================

        # Query all biomarker recordings for selected subjects in the date range
        biomarker_query = db.session.query(
            StudyRecording.recording_id,
            StudyRecording.recording_datetime,
            StudyRecording.subject_id,
            Subject.external_subject_code,
            StudyArm.arm_name,
            BiomarkerType.sample_type,
            BiomarkerType.biomarker_name,
            StudyRecordingBiomarker.biomarker_value
        ).join(
            StudyRecordingBiomarker, StudyRecording.recording_id == StudyRecordingBiomarker.recording_id
        ).join(
            BiomarkerType, StudyRecordingBiomarker.biomarker_id == BiomarkerType.biomarker_id
        ).join(
            Subject, StudyRecording.subject_id == Subject.subject_id
        ).outerjoin(
            StudyArm, Subject.arm_id == StudyArm.arm_id
        ).filter(
            StudyRecording.study_id == study_id,
            StudyRecording.subject_id.in_(subject_ids),
            StudyRecording.recording_type == 'Biomarker',
            StudyRecording.recording_datetime >= start_date,
            StudyRecording.recording_datetime < end_date
        ).order_by(
            Subject.external_subject_code,
            StudyRecording.recording_datetime
        ).all()

        # Build the wide-format DataFrame
        wide_rows = {}  # keyed by (subject_id, recording_datetime)

        for row in biomarker_query:
            # Create a composite key: one row per subject per timepoint
            key = (str(row.subject_id), row.recording_datetime)

            if key not in wide_rows:
                wide_rows[key] = {
                    'Arm': row.arm_name or '',
                    'Subject': row.external_subject_code or str(row.subject_id),
                    'DateTime': row.recording_datetime.isoformat() if row.recording_datetime else '',
                }

            # Build the column name from sample_type and biomarker_name
            # Convention: if sample_type is present → "{SampleType}_{BiomarkerName}"
            #             if sample_type is NULL/empty → "{BiomarkerName}" alone
            sample_type = (row.sample_type or '').strip()
            biomarker_name = (row.biomarker_name or 'Unknown').strip()

            if sample_type:
                column_name = f"{sample_type.replace(' ', '_')}_{biomarker_name.replace(' ', '_')}"
            else:
                column_name = biomarker_name.replace(' ', '_')

            wide_rows[key][column_name] = float(row.biomarker_value) if row.biomarker_value is not None else None

        # =================================================================
        # STEP 2: Query non-biomarker recordings for raw file downloads
        # =================================================================

        file_query = db.session.query(
            StudyRecording.recording_id,
            StudyRecording.recording_datetime,
            StudyRecording.recording_type,
            StudyRecording.subject_id,
            Subject.external_subject_code,
            StudyRecordingEEG.data_uri.label('eeg_uri'),
            StudyRecordingWearable.data_uri.label('wearable_uri'),
            StudyRecordingImage.data_uri.label('image_uri')
        ).join(
            Subject, StudyRecording.subject_id == Subject.subject_id
        ).outerjoin(
            StudyRecordingEEG, StudyRecording.recording_id == StudyRecordingEEG.recording_id
        ).outerjoin(
            StudyRecordingWearable, StudyRecording.recording_id == StudyRecordingWearable.recording_id
        ).outerjoin(
            StudyRecordingImage, StudyRecording.recording_id == StudyRecordingImage.recording_id
        ).filter(
            StudyRecording.study_id == study_id,
            StudyRecording.subject_id.in_(subject_ids),
            StudyRecording.recording_type.in_(['EEG', 'Wearable', 'Imaging']),
            StudyRecording.recording_datetime >= start_date,
            StudyRecording.recording_datetime < end_date
        ).order_by(
            Subject.external_subject_code,
            StudyRecording.recording_datetime
        ).all()

        # =================================================================
        # STEP 3: Assemble the ZIP
        # =================================================================

        memory_file = BytesIO()
        zf = zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED)

        # --- Write the wide-format biomarkers CSV ---
        if wide_rows:
            df_wide = pd.DataFrame(wide_rows.values())

            # Fixed columns come first in exact template order, then biomarker columns sorted
            fixed_cols = ['Arm', 'Subject', 'DateTime']
            biomarker_cols = sorted([c for c in df_wide.columns if c not in fixed_cols])
            df_wide = df_wide[fixed_cols + biomarker_cols]

            zf.writestr('biomarkers.csv', df_wide.to_csv(index=False, encoding='utf-8'))

        # --- Download raw files organised by subject/date ---
        for row in file_query:
            subject_folder = row.external_subject_code or f"Subject_ID_{row.subject_id}"
            date_folder = row.recording_datetime.date().strftime('%Y-%m-%d') if row.recording_datetime else 'unknown_date'
            base_zip_path = f"{subject_folder}/{date_folder}"
            recording_id_str = str(row.recording_id)

            if row.recording_type == 'EEG' and row.eeg_uri:
                zip_path = f"{base_zip_path}/eeg/{recording_id_str}.zip"
                _download_blob_to_zip(zf, row.eeg_uri, zip_path, blob_service_client)

            elif row.recording_type == 'Wearable' and row.wearable_uri:
                zip_path = f"{base_zip_path}/wearable/{recording_id_str}.zip"
                _download_blob_to_zip(zf, row.wearable_uri, zip_path, blob_service_client)

            elif row.recording_type == 'Imaging' and row.image_uri:
                try:
                    path = urlparse(row.image_uri).path
                    filename = os.path.basename(path)
                    _, ext = os.path.splitext(filename)
                    if not ext:
                        ext = ".dat"
                except Exception:
                    ext = ".dat"
                zip_path = f"{base_zip_path}/imaging/{recording_id_str}{ext}"
                _download_blob_to_zip(zf, row.image_uri, zip_path, blob_service_client)

        zf.close()
        memory_file.seek(0)
        filename_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"{study.name.replace(' ', '_')}_Analysis_Packet_{filename_date}.zip"
        )

    except Exception as e:
        app.logger.error(f"Failed to generate multimodal packet for study {study_id}: {e}", exc_info=True)
        flash(f'An error occurred while generating the packet: {e}', 'danger')
        return redirect(url_for('study_recordings', study_id=study_id))


@app.route('/api/study/<study_id>/knowledge_files', methods=['GET'])
@login_required
def get_knowledge_files(study_id):
    try:
        files_query = db.session.query(
            StudyKnowledge.knowledge_id,
            StudyKnowledge.filename,
            StudyKnowledge.upload_date
        ).filter_by(study_id=study_id).order_by(StudyKnowledge.filename)
        
        files = files_query.all()
        files_list = [
            {
                'knowledge_id': str(f.knowledge_id),
                'filename': f.filename,
                'upload_date': f.upload_date.isoformat(),
            } for f in files
        ]
        return jsonify(files_list)
    except Exception as e:
        app.logger.error(f"Error fetching knowledge files for study {study_id}: {e}")
        return jsonify({'error': 'Failed to retrieve knowledge files.'}), 500


@app.route('/api/study/<study_id>/knowledge_upload', methods=['POST'])
@login_required
def upload_knowledge_files(study_id):
    """Handles PDF file uploads for the knowledge base."""
    _require_study_membership(study_id)
    if 'files' not in request.files:
        return jsonify({'message': 'No file part in request.'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'message': 'No files selected.'}), 400

    uploaded_count = 0
    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            try:
                new_doc = StudyKnowledge(
                    study_id=study_id,
                    filename=file.filename,
                    data=file.read()
                )
                db.session.add(new_doc)
                uploaded_count += 1
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Failed to upload knowledge file {file.filename}: {e}")
                return jsonify({'message': f'Error uploading file {file.filename}: {e}'}), 500
    
    try:
        db.session.commit()
        return jsonify({
            'status': 'success', 
            'message': f'{uploaded_count} PDF(s) uploaded successfully. Please "Build" the vector store to use them.'
        })
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to commit knowledge uploads for study {study_id}: {e}")
        return jsonify({'message': 'An internal error occurred during commit. Please try again.'}), 500


@app.route('/api/knowledge/<knowledge_id>', methods=['DELETE'])
@login_required
def delete_knowledge_file(knowledge_id):
    """Deletes a knowledge file and its associated vectors."""
    try:
        doc = StudyKnowledge.query.get(knowledge_id)
        if not doc:
            return jsonify({'message': 'File not found.'}), 404

        StudyKnowledgeVector.query.filter_by(knowledge_id=knowledge_id).delete(synchronize_session=False)

        db.session.delete(doc)
        db.session.commit()
        return jsonify({'status': 'success', 'message': f'File "{doc.filename}" and its vectors have been deleted.'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting knowledge file {knowledge_id}: {e}")
        return jsonify({'message': 'Error deleting file.'}), 500


def get_embedding_client(study_id):
    """Helper to initialize the OpenAI Embeddings client using the API key from study settings."""
    try:
        study_settings = StudySettings.query.get(study_id)
        if not study_settings or not study_settings.ai_enabled or not study_settings.ai_api_key:
            app.logger.error(f"AI features or API key not configured for study {study_id}.")
            return None
        
        api_key = study_settings.ai_api_key
        app.logger.info(f"Attempting to initialize OpenAI client for study {study_id}...")
        client = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        app.logger.info("OpenAI client initialized successfully.")
        return client

    except Exception as e:
        app.logger.error(f"Failed to initialize OpenAIEmbeddings for study {study_id}: {e}")
        return None


@app.route('/api/study/<study_id>/build_knowledge_vectors', methods=['POST'])
@login_required
def build_knowledge_vectors(study_id):
    """
    Builds the vector store incrementally.
    Only processes files that do not currently have vectors.
    Commits after each successful file.
    """
    _require_study_membership(study_id)
    app.logger.info(f"Incremental knowledge build route HIT for study {study_id}.")
    study = Study.query.get_or_404(study_id)
    study_settings = study.settings
    if not study_settings or not study_settings.ai_enabled or not study_settings.ai_api_key:
        return jsonify({'message': 'AI features or API key not configured for this study.'}), 400

    embeddings_client = get_embedding_client(study_id)
    if not embeddings_client:
        return jsonify({'message': 'Failed to initialize AI embeddings service. Check API key.'}), 500

    app.logger.info("Fetching all knowledge files...")
    all_knowledge_files = StudyKnowledge.query.filter_by(study_id=study_id).all()
    if not all_knowledge_files:
        return jsonify({'message': 'No knowledge files found to build.'}), 400

    app.logger.info("Checking for already processed files...")
    existing_vector_file_ids = db.session.query(
        StudyKnowledgeVector.knowledge_id
    ).distinct().join(
        StudyKnowledge, StudyKnowledge.knowledge_id == StudyKnowledgeVector.knowledge_id
    ).filter(
        StudyKnowledge.study_id == study_id
    ).all()
    processed_file_ids = {str(id[0]) for id in existing_vector_file_ids}
    app.logger.info(f"Found {len(processed_file_ids)} already processed files.")

    files_to_process = [
        f for f in all_knowledge_files 
        if str(f.knowledge_id) not in processed_file_ids
    ]

    if not files_to_process:
        app.logger.info("No new files to process. Vector store is up to date.")
        return jsonify({
            'status': 'success',
            'message': 'Build complete: All files were already processed.'
        })

    app.logger.info(f"Found {len(files_to_process)} new files to process. Starting loop...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    total_chunks = 0
    total_files_processed = 0

    for file_obj in files_to_process: # Loop over the *filtered* list
        app.logger.info(f"Processing new knowledge file: {file_obj.filename}")
        try:
            StudyKnowledgeVector.query.filter_by(knowledge_id=file_obj.knowledge_id).delete(synchronize_session=False)

            pdf_bytes = BytesIO(file_obj.data)
            reader = PdfReader(pdf_bytes)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            
            if not full_text.strip():
                app.logger.warning(f"Skipping empty file: {file_obj.filename}")
                continue

            chunks = text_splitter.split_text(full_text)
            if not chunks:
                app.logger.warning(f"No text chunks extracted from: {file_obj.filename}")
                continue
            
            app.logger.info(f"Embedding {len(chunks)} chunks for {file_obj.filename}...")
            vectors = embeddings_client.embed_documents(chunks)
            app.logger.info(f"Embedding complete for {file_obj.filename}.")

            for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                new_vector_entry = StudyKnowledgeVector(
                    knowledge_id=file_obj.knowledge_id,
                    chunk_index=i,
                    text_content_preview=chunk_text,
                    vector_data=json.dumps(vector).encode('utf-8')
                )
                db.session.add(new_vector_entry)
                total_chunks += 1
            
            db.session.commit()
            total_files_processed += 1
            app.logger.info(f"Successfully processed and committed file: {file_obj.filename}")

        except Exception as e_file:
            db.session.rollback()
            app.logger.error(f"Failed to process file {file_obj.filename}: {e_file}. Skipping to next file.")

    app.logger.info("Build loop finished.")
    return jsonify({
        'status': 'success',
        'message': f'Build complete: {total_chunks} new vectors created from {total_files_processed} file(s).'
    })


def _get_rag_context(study_id, query_text, max_context_items=10, min_similarity=0.65):
    """
    Helper function to perform RAG:
    1. Get all vector chunks for the study.
    2. Create a query vector from biomarker names.
    3. Find top_k most similar chunks.
    4. Return the text of those chunks.
    """
    try:
        chunks = db.session.query(StudyKnowledgeVector.text_content_preview, StudyKnowledgeVector.vector_data)\
            .join(StudyKnowledge, StudyKnowledge.knowledge_id == StudyKnowledgeVector.knowledge_id)\
            .filter(StudyKnowledge.study_id == study_id)\
            .all()
        
        if not chunks:
            app.logger.info("No knowledge vectors found for RAG.")
            return ""

        db_texts = []
        db_vectors = []
        for c in chunks:
            try:
                raw = bytes(c.vector_data) if not isinstance(c.vector_data, bytes) else c.vector_data
                db_vectors.append(json.loads(raw.decode('utf-8')))
                db_texts.append(c.text_content_preview)
            except Exception:
                app.logger.warning("Skipping knowledge vector with unreadable encoding; rebuild required.")
                continue
        
        embeddings_client = get_embedding_client(study_id)
        if not embeddings_client:
            app.logger.warning(f"Could not get embedding client for RAG in study {study_id}.")
            return ""
        
        query_vector = embeddings_client.embed_query(query_text)
        query_vector_np = np.array(query_vector).reshape(1, -1)
        db_vectors_np = np.array(db_vectors)

        if db_vectors_np.ndim == 1:
            db_vectors_np = db_vectors_np.reshape(1, -1)

        similarities = cosine_similarity(query_vector_np, db_vectors_np)[0]
        top_indices = np.argsort(similarities)[-max_context_items:][::-1]
        context_items = []
        for i in top_indices:
            if similarities[i] >= min_similarity:
                context_items.append(db_texts[i])
            else:
                break
        
        if not context_items:
            app.logger.info("RAG: No chunks met similarity threshold.")
            return ""

        context_text = "\n\n---\n\n".join(context_items)
        app.logger.info(f"RAG: Found {len(context_items)} relevant context chunks.")
        return context_text

    except Exception as e:
        app.logger.error(f"Error during RAG context retrieval: {e}")
        return ""


@app.route('/api/meddra/search', methods=['GET'])
@login_required
def meddra_search():
    """
    API endpoint for the MedDRA search box, as seen in
    study_recordings.html.
    """
    query = request.args.get('q', '')
    if len(query) < 3:
        return jsonify([])

    search_term = f"%{query}%"
    
    # Search on term, low level term, or system organ class
    results = MedDRA.query.filter(
        or_(
            MedDRA.meddra_term.ilike(search_term),
            MedDRA.llt_name.ilike(search_term),
            MedDRA.soc_name.ilike(search_term)
        )
    ).limit(20).all()

    return jsonify([
        {
            'meddra_code': r.meddra_code,
            'meddra_term': r.meddra_term,
            'llt_name': r.llt_name,
            'soc_name': r.soc_name
        } for r in results
    ])


@app.route('/api/research', methods=['POST'])
@login_required
def research_query():
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        api_key = None

        # First, try API keys from studies the current user belongs to
        participant = StudyParticipant.query.filter_by(email=current_user.email).first()
        if participant:
            user_study_ids = db.session.query(StudyParticipantLink.study_id).filter_by(
                participant_id=str(participant.participant_id).lower()
            ).subquery()
            study_settings = StudySettings.query.filter(
                StudySettings.study_id.in_(user_study_ids),
                StudySettings.ai_api_key != None,
            ).first()
            if study_settings:
                api_key = study_settings.ai_api_key

        # Fallback to current user's personal API key
        if not api_key and current_user.ai_api_key:
            api_key = current_user.ai_api_key

        # If still no key, return specific error for modal
        if not api_key:
            return jsonify({
                'error': 'API_KEY_MISSING',
                'message': 'Please configure an Open-AI API Key on either the Study Settings or your User Account.'
            }), 400

        client = openai.OpenAI(api_key=api_key)

        extraction_prompt = f"""
        Extract search parameters for ClinicalTrials.gov from this user query:
        "{user_query}"

        Return ONLY a raw JSON object (no markdown) with keys:
        - "condition": string or null (use common ClinicalTrials.gov condition names,
          e.g. "Diabetes Mellitus" instead of "diabetes")
        - "intervention": string or null
        - "status": string (ENUM: RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING) or null
        - "metric": string (ENUM: enrollment, duration)

        If the user asks about typical, average, or historical trends,
        prefer COMPLETED studies.
        """

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0
        )

        params = json.loads(completion.choices[0].message.content)

        CONDITION_SYNONYMS = {
            "diabetes": [
                "Diabetes Mellitus",
                "Type 1 Diabetes Mellitus",
                "Type 2 Diabetes Mellitus"
            ]
        }

        raw_condition = params.get('condition')
        conditions = []

        if raw_condition:
            key = raw_condition.lower()
            conditions = CONDITION_SYNONYMS.get(key, [raw_condition])

        if params.get('metric') == 'duration' and not params.get('status'):
            params['status'] = 'COMPLETED'

        ct_service = ClinicalTrialsService()

        df = ct_service.search_studies(
            condition=conditions,
            intervention=params.get('intervention'),
            status=params.get('status'),
            limit=300
        )

        if df.empty:
            return jsonify({
                'summary': "No matching studies found on ClinicalTrials.gov with the provided criteria.",
                'plot': None
            })

        def remove_outliers(d, col):
            q1 = d[col].quantile(0.25)
            q3 = d[col].quantile(0.75)
            iqr = q3 - q1
            return d[(d[col] >= q1 - 1.5 * iqr) & (d[col] <= q3 + 1.5 * iqr)]

        metric = params.get('metric', 'enrollment')
        plot_json = None

        if metric == 'duration':
            df_duration = df.dropna(subset=['duration_months'])

            if df_duration.empty:
                return jsonify({
                    'summary': "Relevant studies were found, but duration data was insufficient for analysis.",
                    'plot': None
                })

            df_clean = remove_outliers(df_duration, 'duration_months')

            stats_summary = (
                f"analyzed {len(df_clean)} completed studies. "
                f"The median duration is {df_clean['duration_months'].median():.1f} months."
            )

            hist, edges = np.histogram(df_clean['duration_months'], bins=20)

            p = figure(
                title=f"Distribution of Study Duration ({raw_condition or 'Search'})",
                height=300,
                sizing_mode='stretch_width',
                tools=""
            )

            p.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                fill_color="#20519c",
                line_color="white",
                alpha=0.8
            )

            p.xaxis.axis_label = "Duration (Months)"
            p.yaxis.axis_label = "Frequency"

            plot_json = json.dumps(json_item(p, "researchPlotContainer"))

        else:
            df_enroll = df.dropna(subset=['enrollment'])

            if df_enroll.empty:
                return jsonify({
                    'summary': "Relevant studies were found, but enrollment data was insufficient for analysis.",
                    'plot': None
                })

            df_clean = remove_outliers(df_enroll, 'enrollment')

            stats_summary = (
                f"analyzed {len(df_clean)} studies. "
                f"The median enrollment is {df_clean['enrollment'].median():.0f} participants."
            )

            p = figure(
                title=f"Enrollment Sizes ({raw_condition or 'Search'})",
                height=300,
                sizing_mode='stretch_width',
                tools="hover"
            )

            x_vals = _rng.normal(0.5, 0.05, len(df_clean))
            p.circle(x_vals, df_clean['enrollment'], size=8, color="#20519c", alpha=0.6)

            p.xaxis.visible = False
            p.yaxis.axis_label = "Number of Participants"

            plot_json = json.dumps(json_item(p, "researchPlotContainer"))

        summary_prompt = f"""
        User asked: "{user_query}"

        Data findings: I {stats_summary}

        Provide a concise 2–3 sentence professional summary answering the user's question.
        """

        summary_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}]
        )

        return jsonify({
            'summary': summary_completion.choices[0].message.content,
            'plot': plot_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/visualize_image/<recording_id>', methods=['GET'])
@login_required
def visualize_image(recording_id):
    """
    Renders the Imaging visualization page (Static PNG preview).
    """
    recording = StudyRecording.query.get_or_404(recording_id)
    subject = Subject.query.get_or_404(recording.subject_id)
    study = Study.query.get_or_404(str(recording.study_id))
    
    participant = StudyParticipant.query.filter_by(email=current_user.email).first()
    if not participant:
         flash('You must be a participant in this study to view recordings.', 'danger')
         return redirect(url_for('main_app'))
    
    link = StudyParticipantLink.query.filter_by(
        study_id=study.study_id, 
        participant_id=participant.participant_id
    ).first()
    
    if not link:
        flash('You are not a participant in this study.', 'danger')
        return redirect(url_for('main_app'))

    img_data, error_message = _load_and_process_image(recording_id)

    if error_message:
        flash(f"Could not visualize Image: {error_message}", 'danger')
        return redirect(url_for('study_recordings', study_id=study.study_id))
    
    image_record = StudyRecordingImage.query.get(recording_id)
    image_type = image_record.image_type if image_record else "Unknown"

    return render_template(
        'visualize_image.html',
        study=study,
        subject=subject,
        recording=recording,
        image_data=img_data,
        image_type=image_type
    )


@app.errorhandler(403)
def forbidden(e):
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'error': 'Forbidden', 'message': 'You do not have permission to access this resource.'}), 403
    flash('You do not have permission to access that resource.', 'danger')
    return redirect(url_for('main_app')), 403


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'}), 500
    return render_template('base.html'), 500


if __name__ == '__main__':
    if not os.getenv("WEBSITE_HOSTNAME"):
        load_dotenv()
        with app.app_context():
            db.create_all()
        app.run(host="0.0.0.0", port=8000, debug=True)