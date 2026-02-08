import uuid
from datetime import datetime
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy import UniqueConstraint
import uuid
from extensions import db, login_manager
from sqlalchemy import LargeBinary
from sqlalchemy import event
from sqlalchemy.orm.attributes import get_history
from sqlalchemy.orm import class_mapper
from sqlalchemy.orm.exc import UnmappedClassError
from flask import g
import json


class User(UserMixin, db.Model):
    """User model for the database."""
    id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    email_confirmed = db.Column(db.Boolean, nullable=False, default=False)
    ai_api_key = db.Column(db.String(250), nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login user loader callback."""
    return User.query.get(user_id)


class Study(db.Model):
    """
    Represents a research study.
    """
    __tablename__ = 'studies'
    
    study_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(50), nullable=True)
    funding_source = db.Column(db.String(255), nullable=True)
    budget_amount = db.Column(db.Numeric(18, 2), nullable=True)
    currency = db.Column(db.String(10), nullable=True)
    principal_investigator_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    study_type_id = db.Column(db.String(36), db.ForeignKey('study_types.study_type_id'), nullable=True)

    participants = db.relationship('StudyParticipantLink', backref='study', lazy=True)
    documents = db.relationship('StudyDocument', backref='study', lazy=True, cascade="all, delete-orphan")
    knowledge_files = db.relationship('StudyKnowledge', backref='study', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Study {self.name}>'


class StudyParticipantLink(db.Model):
    """
    Association table linking users (participants) to studies.
    """
    __tablename__ = 'study_participant_link'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    study_id = db.Column(db.String(36), db.ForeignKey('studies.study_id'), nullable=False)
    participant_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    active = db.Column(db.Boolean, default=True, nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

    def __repr__(self):
        return f'<StudyParticipantLink {self.participant_id} in {self.study_id}>'


class StudyType(db.Model):
    """
    Represents the different types of studies available.
    """
    __tablename__ = 'study_types'

    study_type_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    category = db.Column(db.String(100), nullable=False)
    study_type = db.Column(db.String(100), nullable=False)
    
    studies = db.relationship('Study', backref='study_type', lazy=True)

    def __repr__(self):
        return f'<StudyType {self.category} - {self.study_type}>'


class StudyParticipant(db.Model):
    __tablename__ = 'study_participants'
    participant_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    email = db.Column(db.String(150), unique=True)
    phone = db.Column(db.String(50))
    role = db.Column(db.String(100))
    affiliation = db.Column(db.String(255))
    department = db.Column(db.String(150))
    orcid = db.Column(db.String(50))
    country = db.Column(db.String(100))
    funding_role = db.Column(db.String(150))
    percent_effort = db.Column(db.Numeric(5, 2))
    salary_contribution = db.Column(db.Numeric(18, 2))

    def __repr__(self):
        return f'<StudyParticipant {self.first_name} {self.last_name}>'
  
    
class StudyArm(db.Model):
    """Represents the actual participant arms for a specific study instance."""
    __tablename__ = 'study_arms'
    arm_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    arm_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<StudyArm {self.arm_name}>'
    

class Subject(db.Model):
    """Represents a single subject or participant in a study."""
    __tablename__ = 'subjects'
    
    subject_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    external_subject_code = db.Column(db.String(50), nullable=True)
    study_id = db.Column(db.String(36), db.ForeignKey('studies.study_id'), nullable=True)
    arm_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_arms.arm_id'), nullable=True)
    status = db.Column(db.String(50), nullable=True, default='Screening') # e.g., Screening, Enrolled, Completed, Withdrawn, Screen Fail
    enrollment_date = db.Column(db.Date, nullable=True)
    completion_date = db.Column(db.Date, nullable=True)
    screen_fail_date = db.Column(db.Date, nullable=True)
    screen_fail_reason = db.Column(db.Text, nullable=True)
    withdrawal_reason = db.Column(db.Text, nullable=True)
    site_identifier = db.Column(db.String(100), nullable=True)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    ethnicity = db.Column(db.String(100), nullable=True)
    race = db.Column(db.String(100), nullable=True)
    handedness = db.Column(db.String(20), nullable=True)
    pregnancy_status = db.Column(db.String(50), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    state_province = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    postal_code = db.Column(db.String(20), nullable=True)
    education_level = db.Column(db.String(100), nullable=True)
    employment_status = db.Column(db.String(100), nullable=True)
    marital_status = db.Column(db.String(50), nullable=True)
    height_cm = db.Column(db.Numeric(5, 2), nullable=True)
    weight_kg = db.Column(db.Numeric(5, 2), nullable=True)
    smoking_status = db.Column(db.String(20), nullable=True)
    alcohol_intake = db.Column(db.String(50), nullable=True)
    physical_activity_level = db.Column(db.String(20), nullable=True)
    consent_date = db.Column(db.Date, nullable=True)
    withdrawal_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    clinicians = db.relationship('SubjectClinician', backref='subject', lazy=True, cascade="all, delete-orphan")
    consents = db.relationship('SubjectConsent', backref='subject', lazy=True, cascade="all, delete-orphan")
    contacts = db.relationship('SubjectContact', backref='subject', lazy=True, cascade="all, delete-orphan")
    diagnoses = db.relationship('SubjectDiagnosis', backref='subject', lazy=True, cascade="all, delete-orphan")
    medications = db.relationship('SubjectMedication', backref='subject', lazy=True, cascade="all, delete-orphan")
    documents = db.relationship('SubjectDocument', backref='subject', lazy=True, cascade="all, delete-orphan")
    arm = db.relationship('StudyArm', backref='subjects')

    __table_args__ = (UniqueConstraint('study_id', 'external_subject_code', name='_study_subject_uc'),)

    def __repr__(self):
        return f'<Subject {self.first_name} {self.last_name}>'


class SubjectClinician(db.Model):
    """Represents clinicians associated with a subject."""
    __tablename__ = 'subject_clinicians'

    clinician_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    specialty = db.Column(db.String(100), nullable=True)
    organization = db.Column(db.String(150), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    country = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(150), nullable=True)
    phone = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f'<SubjectClinician {self.first_name} {self.last_name}>'


class SubjectConsent(db.Model):
    """Represents consent forms and status for a subject."""
    __tablename__ = 'subject_consent'

    consent_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    consent_version = db.Column(db.String(50), nullable=True)
    consent_type = db.Column(db.String(50), nullable=True)
    signed_at = db.Column(db.DateTime, nullable=True)
    withdrawn_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<SubjectConsent {self.consent_id} for Subject {self.subject_id}>'


class SubjectContact(db.Model):
    """Represents contact information for a subject."""
    __tablename__ = 'subject_contacts'

    contact_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    contact_type = db.Column(db.String(20), nullable=True)
    contact_value = db.Column(db.String(255), nullable=True)
    preferred = db.Column(db.Boolean, nullable=True)
    verified = db.Column(db.Boolean, nullable=True)

    def __repr__(self):
        return f'<SubjectContact {self.contact_type}: {self.contact_value}>'


class SubjectDiagnosis(db.Model):
    """Represents a subject's medical diagnoses."""
    __tablename__ = 'subject_diagnoses'

    id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    diagnosis_code = db.Column(db.String(20), nullable=True)
    diagnosis_description = db.Column(db.String(255), nullable=True)
    diagnosis_date = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(20), nullable=True)
    primary_diagnosis = db.Column(db.Boolean, nullable=True)

    def __repr__(self):
        return f'<SubjectDiagnosis {self.diagnosis_code}>'


class SubjectMedication(db.Model):
    """Represents medications taken by a subject (medical history)."""
    __tablename__ = 'subject_medications'

    id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    medication_name = db.Column(db.String(150), nullable=False)
    dose = db.Column(db.String(100), nullable=True)
    route = db.Column(db.String(50), nullable=True)
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)
    indication = db.Column(db.String(150), nullable=True)
    currently_taking = db.Column(db.Boolean, nullable=True)

    def __repr__(self):
        return f'<SubjectMedication {self.medication_name}>'
    

class StudyDocument(db.Model):
    """Represents a document attached to a study."""
    __tablename__ = 'study_documents'
    
    document_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    study_id = db.Column(db.String(36), db.ForeignKey('studies.study_id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    data = db.Column(LargeBinary, nullable=False) # Stores the file content

    def __repr__(self):
        return f'<StudyDocument {self.filename}>'
    

class SubjectDocument(db.Model):
    """Represents a document attached to a subject."""
    __tablename__ = 'subject_documents'
    
    document_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f'<SubjectDocument {self.filename}>'
    

class StudyRecording(db.Model):
    """Represents a recording session for a study subject."""
    __tablename__ = 'study_recordings'

    recording_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    recording_datetime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    recording_type = db.Column(db.String(10), nullable=False) # e.g., 'Biomarker', 'EEG', 'Wearable', 'Symptom', 'Medication', 'Adverse Event'

    subject = db.relationship('Subject', backref=db.backref('recordings', cascade="all, delete-orphan"))
    study = db.relationship('Study', backref=db.backref('recordings', cascade="all, delete-orphan"))
    biomarker_data = db.relationship('StudyRecordingBiomarker', backref='recording', uselist=False, cascade="all, delete-orphan")
    eeg_data = db.relationship('StudyRecordingEEG', backref='recording', uselist=False, cascade="all, delete-orphan")
    wearable_data = db.relationship('StudyRecordingWearable', backref='recording', uselist=False, cascade="all, delete-orphan")
    symptom_data = db.relationship('SubjectSymptom', backref='recording', uselist=False, cascade="all, delete-orphan")
    medication_taken_data = db.relationship('SubjectMedicationTaken', backref='recording', uselist=False, cascade="all, delete-orphan")
    adverse_event_data = db.relationship('SubjectAdverseEvent', backref='recording', uselist=False, cascade="all, delete-orphan")
    image_data = db.relationship('StudyRecordingImage', backref='recording', uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<StudyRecording {self.recording_id}>'
    

class BiomarkerType(db.Model):
    """Represents the types of biomarkers that can be recorded."""
    __tablename__ = 'biomarker_types'

    biomarker_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    sample_type = db.Column(db.String(100))
    biomarker_name = db.Column(db.String(150), nullable=False)
    category_notes = db.Column(db.Text)

    def __repr__(self):
        return f'<BiomarkerType {self.biomarker_name}>'


class StudyRecordingBiomarker(db.Model):
    """Stores the specific value for a biomarker recording."""
    __tablename__ = 'study_recordings_biomarker'

    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    biomarker_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('biomarker_types.biomarker_id'), nullable=False)
    biomarker_value = db.Column(db.Numeric(18, 5), nullable=False)

    biomarker_type = db.relationship('BiomarkerType')
    
    def __repr__(self):
        return f'<StudyRecordingBiomarker for Recording {self.recording_id}>'
    

class EEG(db.Model):
    """Represents an EEG device."""
    __tablename__ = 'eeg'
    eeg_id = db.Column(db.Integer, primary_key=True)
    manufacturer = db.Column(db.String(255), nullable=False)
    device_type = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<EEG {self.manufacturer} {self.device_type}>'


class Wearable(db.Model):
    """Represents a wearable device."""
    __tablename__ = 'wearables'
    wearable_id = db.Column(db.Integer, primary_key=True)
    manufacturer = db.Column(db.String(255), nullable=False)
    device_name = db.Column(db.String(255), nullable=False)
    wearable_type = db.Column(db.String(255), nullable=True)
    wearable_location = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f'<Wearable {self.manufacturer} {self.device_name}>'
    

class StudyRecordingEEG(db.Model):
    """Stores the data URI for an EEG recording in Azure Blob Storage."""
    __tablename__ = 'study_recordings_eeg'

    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    data_uri = db.Column(db.String(250), nullable=False)
    eeg_id = db.Column(db.Integer, db.ForeignKey('eeg.eeg_id'), nullable=True)

    eeg_device = db.relationship('EEG')

    def __repr__(self):
        return f'<StudyRecordingEEG for Recording {self.recording_id}>'


class StudyRecordingWearable(db.Model):
    """Stores the data URI for a Wearable recording in Azure Blob Storage."""
    __tablename__ = 'study_recordings_wearable'

    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    data_uri = db.Column(db.String(250), nullable=False)
    wearable_id = db.Column(db.Integer, db.ForeignKey('wearables.wearable_id'), nullable=True)

    wearable_device = db.relationship('Wearable')

    def __repr__(self):
        return f'<StudyRecordingWearable for Recording {self.recording_id}>'
    

class StudyRecordingImage(db.Model):
    """Stores the data URI for a Image recording in Azure Blob Storage."""
    __tablename__ = 'study_recordings_image'

    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    data_uri = db.Column(db.String(250), nullable=False)
    image_type = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<StudyRecordingImage for Recording {self.recording_id}>'
        

class ExpenseCategory(db.Model):
    """Represents a category for a financial expense."""
    __tablename__ = 'expense_categories'
    
    category_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    category_name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<ExpenseCategory {self.category_name}>'


class FinancialLedger(db.Model):
    """Represents a single transaction in the financial ledger for a study."""
    __tablename__ = 'financial_ledger'

    transaction_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    transaction_date = db.Column(db.Date, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # 'BUDGET', 'TOPUP', 'EXPENSE'
    amount = db.Column(db.Numeric(18, 2), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('expense_categories.category_id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    study = db.relationship('Study', backref=db.backref('financial_transactions', cascade="all, delete-orphan"))
    category = db.relationship('ExpenseCategory', backref='ledger_entries')

    def __repr__(self):
        return f'<FinancialLedger {self.transaction_id} for Study {self.study_id}>'
    

study_settings_biomarker_types = db.Table('study_settings_biomarker_types',
    db.Column('study_id', UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id', ondelete='CASCADE'), primary_key=True),
    db.Column('biomarker_id', UNIQUEIDENTIFIER, db.ForeignKey('biomarker_types.biomarker_id', ondelete='CASCADE'), primary_key=True)
)


class StudySettings(db.Model):
    """
    Represents study-specific settings, in a one-to-one relationship with a Study.
    """
    __tablename__ = 'study_settings'
    
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id', ondelete='CASCADE'), primary_key=True)
    ai_enabled = db.Column(db.Boolean, nullable=False, default=False)
    ai_api_key = db.Column(db.String(250), nullable=True)
    eeg_enabled = db.Column(db.Boolean, nullable=False, default=False)
    wearables_enabled = db.Column(db.Boolean, nullable=False, default=False)
    biological_enabled = db.Column(db.Boolean, nullable=False, default=False)
    scales_enabled = db.Column(db.Boolean, default=False)
    
    study = db.relationship('Study', backref=db.backref('settings', uselist=False, cascade="all, delete-orphan"))
    
    allowed_biomarkers = db.relationship(
        'BiomarkerType', 
        secondary=study_settings_biomarker_types, 
        lazy='subquery', 
        backref=db.backref('studies_settings', lazy=True),
        primaryjoin="StudySettings.study_id == study_settings_biomarker_types.c.study_id",
        secondaryjoin="BiomarkerType.biomarker_id == study_settings_biomarker_types.c.biomarker_id"
    )

    def __repr__(self):
        return f'<StudySettings for Study {self.study_id}>'


class StudyKnowledge(db.Model):
    """Stores uploaded PDF knowledge files for a study."""
    __tablename__ = 'studies_knowledge'

    knowledge_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    data = db.Column(LargeBinary, nullable=False) # Stores the PDF file content

    vectors = db.relationship('StudyKnowledgeVector', backref='knowledge_file', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<StudyKnowledge {self.filename} for Study {self.study_id}>'


class StudyKnowledgeVector(db.Model):
    """Stores text chunks and vectors from knowledge files."""
    __tablename__ = 'studies_knowledge_vector'

    vector_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    knowledge_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies_knowledge.knowledge_id', ondelete='CASCADE'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    text_content_preview = db.Column(db.Text, nullable=True)
    vector_data = db.Column(LargeBinary, nullable=False)

    __table_args__ = (UniqueConstraint('knowledge_id', 'chunk_index', name='_knowledge_chunk_uc'),)

    def __repr__(selfD):
        return f'<StudyKnowledgeVector Chunk {self.chunk_index} for File {self.knowledge_id}>'


class AuditLog(db.Model):
    """Represents a single audit trail entry for changes to recordings."""
    __tablename__ = 'audit_log'
    
    audit_log_id = db.Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)
    change_datetime = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_email = db.Column(db.String(120), nullable=False)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=True)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=True)
    record_id = db.Column(UNIQUEIDENTIFIER, nullable=False)
    changed_table = db.Column(db.String(128), nullable=False)
    change_type = db.Column(db.String(50), nullable=False) # 'Wearable', 'EEG', 'Biomarker'
    operation_type = db.Column(db.String(10), nullable=False) # 'INSERT', 'UPDATE', 'DELETE'
    old_value = db.Column(db.Text, nullable=True)
    new_value = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<AuditLog {self.operation_type} on {self.changed_table} by {self.user_email}>'


class MedDRA(db.Model):
    """
    Represents a simplified MedDRA dictionary entry.
    This table is intended to be pre-populated.
    """
    __tablename__ = 'meddra_dictionary'
    
    meddra_code = db.Column(db.String(20), primary_key=True)
    meddra_term = db.Column(db.String(255), nullable=False)
    soc_name = db.Column(db.String(255), nullable=True) # System Organ Class
    hlt_name = db.Column(db.String(255), nullable=True) # High Level Term
    pt_name = db.Column(db.String(255), nullable=True)  # Preferred Term
    llt_name = db.Column(db.String(255), nullable=True) # Low Level Term
    
    def __repr__(self):
        return f'<MedDRA {self.meddra_code} - {self.meddra_term}>'


class SubjectSymptom(db.Model):
    """Stores a single, reported symptom for a subject, linked to a recording."""
    __tablename__ = 'subject_symptoms'
    
    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    
    symptom_verbatim = db.Column(db.Text, nullable=True)
    meddra_code = db.Column(db.String(20), db.ForeignKey('meddra_dictionary.meddra_code'), nullable=True)
    meddra_term = db.Column(db.String(255), nullable=True)
    severity = db.Column(db.String(50), nullable=True)
    
    meddra_entry = db.relationship('MedDRA')

    def __repr__(self):
        return f'<SubjectSymptom {self.meddra_term} for {self.subject_id}>'


class SubjectAdverseEvent(db.Model):
    """Stores a single Adverse Event (AE) for a subject, linked to a recording."""
    __tablename__ = 'subject_adverse_events'
    
    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)
    
    ae_verbatim = db.Column(db.Text, nullable=True) # Clinician's description
    meddra_code = db.Column(db.String(20), db.ForeignKey('meddra_dictionary.meddra_code'), nullable=True)
    meddra_term = db.Column(db.String(255), nullable=True) # The selected MedDRA term
    
    is_serious_ae = db.Column(db.Boolean, nullable=False, default=False)
    severity_grade = db.Column(db.String(50), nullable=True) # e.g., "Grade 1", "Grade 2", "Grade 3"
    causality = db.Column(db.String(50), nullable=True) # Relationship to drug: "Not Related", "Possible", "Probable", etc.
    outcome = db.Column(db.String(100), nullable=True) # e.g., "Resolved", "Ongoing", "Fatal"

    meddra_entry = db.relationship('MedDRA')

    def __repr__(self):
        return f'<SubjectAdverseEvent {self.meddra_term} for {self.subject_id}>'


class SubjectMedicationTaken(db.Model):
    """Stores a log of medication taken (e.g., Concomitant Meds), linked to a recording."""
    __tablename__ = 'subject_medications_taken'
    
    recording_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('study_recordings.recording_id'), primary_key=True)
    study_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('studies.study_id'), nullable=False)
    subject_id = db.Column(UNIQUEIDENTIFIER, db.ForeignKey('subjects.subject_id'), nullable=False)

    medication_name = db.Column(db.String(255), nullable=False)
    dose = db.Column(db.String(100), nullable=True)
    route = db.Column(db.String(50), nullable=True)
    indication = db.Column(db.Text, nullable=True) # Reason for taking
    is_concomitant = db.Column(db.Boolean, nullable=False, default=True)

    def __repr__(self):
        return f'<SubjectMedicationTaken {self.medication_name} for {self.subject_id}>'


def get_change_type_from_target(target):
    """Helper to determine the 'change_type' based on the model class."""
    if isinstance(target, StudyRecording):
        return target.recording_type
    elif isinstance(target, StudyRecordingWearable):
        return "Wearable"
    elif isinstance(target, StudyRecordingEEG):
        return "EEG"
    elif isinstance(target, StudyRecordingBiomarker):
        return "Biomarker"
    elif isinstance(target, SubjectSymptom):
        return "Symptom"
    elif isinstance(target, SubjectAdverseEvent):
        return "Adverse Event"
    elif isinstance(target, SubjectMedicationTaken):
        return "Medication"
    elif isinstance(target, StudyRecordingImage):
        return "Imaging"
    return "Unknown"


def serialize_changes(target, operation_type):
    """Helper to serialize model state for logging."""
    old_values = {}
    new_values = {}

    try:
        inspr = db.inspect(target)
        attrs = [c.key for c in inspr.mapper.column_attrs]
        
        if operation_type == 'INSERT':
            for key in attrs:
                val = getattr(target, key)
                new_values[key] = str(val) if val is not None else None
        elif operation_type == 'DELETE':
            for key in attrs:
                val = getattr(target, key)
                old_values[key] = str(val) if val is not None else None
        elif operation_type == 'UPDATE':
            for key in attrs:
                hist = get_history(target, key)
                if hist.has_changes():
                    old_values[key] = str(hist.deleted[0]) if hist.deleted else None
                    new_values[key] = str(hist.added[0]) if hist.added else None
    except Exception as e:
        # Fallback for serialization errors to prevent transaction blocking
        print(f"Audit Serialization Error: {e}")
        return None, None
                
    return json.dumps(old_values) if old_values else None, \
           json.dumps(new_values) if new_values else None


def get_pk_value(target):
    """Dynamically returns the primary key value and validates if it is a UUID."""
    try:
        mapper = class_mapper(target.__class__)
        primary_key = mapper.primary_key[0] # Assumes single PK
        value = getattr(target, primary_key.name)
        
        # Validation: AuditLog.record_id is UNIQUEIDENTIFIER. 
        # We must skip if the PK is an Integer (e.g. EEG, Wearable tables)
        if isinstance(value, int):
            return None
            
        return value
    except (IndexError, UnmappedClassError):
        return None


def audit_listener(target, operation_type):
    """Generic listener function to create an audit log entry for ANY table."""
    try:
        if g.get('disable_auditing', False): 
            return

        # 1. Check if session exists (detached objects cannot be audited)
        session = db.object_session(target)
        if not session:
            return

        # 2. Get Primary Key (Safety Check)
        record_id = get_pk_value(target)
        if not record_id:
            # Skip tables with Integer PKs or no PK to prevent SQL Type Errors
            return

        # 3. Serialize Data
        old_val, new_val = serialize_changes(target, operation_type)
        
        # Skip updates where nothing actually changed
        if operation_type == 'UPDATE' and not old_val and not new_val:
            return

        # 4. Extract context (user, study, subject) generically
        user_email = g.get('user_email', 'system')
        study_id = getattr(target, 'study_id', None)
        subject_id = getattr(target, 'subject_id', None)
        
        # Change Type is now just the table name to support all tables generically
        # You can keep the helper if you want 'pretty' names, or use __tablename__
        change_type = get_change_type_from_target(target)
        if change_type == "Unknown":
            change_type = target.__tablename__

        log_entry = AuditLog(
            user_email=user_email,
            study_id=study_id,
            subject_id=subject_id,
            record_id=record_id,
            changed_table=target.__tablename__,
            change_type=change_type,
            operation_type=operation_type,
            old_value=old_val,
            new_value=new_val
        )
        session.add(log_entry)
        
    except Exception as e:
        # Fail silently in production logging to not crash the main transaction
        print(f"Error in audit_listener for {target.__tablename__}: {e}")


def audit_insert(mapper, connection, target):
    audit_listener(target, 'INSERT')

def audit_update(mapper, connection, target):
    audit_listener(target, 'UPDATE')

def audit_delete(mapper, connection, target):
    audit_listener(target, 'DELETE')

# --- DYNAMIC REGISTRATION LOOP ---
# Iterate over all models registered with SQLAlchemy and attach listeners

def get_all_models():
    """Helper to find models in both old and new SQLAlchemy versions."""
    # Attempt 1: Newer SQLAlchemy (1.4+) location
    if hasattr(db.Model, 'registry') and hasattr(db.Model.registry, '_class_registry'):
        return db.Model.registry._class_registry.values()
    
    # Attempt 2: Older SQLAlchemy (<1.4) location
    if hasattr(db.Model, '_decl_class_registry'):
        return db.Model._decl_class_registry.values()
        
    # Attempt 3: Direct subclasses (fallback)
    return db.Model.__subclasses__()

registry_items = get_all_models()

for cls in registry_items:
    # Ensure it's a valid SQLAlchemy model class (filters out internal registry artifacts)
    if not isinstance(cls, type) or not issubclass(cls, db.Model):
        continue
        
    # 1. Do not audit the AuditLog itself (infinite recursion)
    if hasattr(cls, '__tablename__') and cls.__tablename__ == 'audit_log':
        continue
        
    # 2. Attach Listeners
    event.listen(cls, 'after_insert', audit_insert)
    event.listen(cls, 'after_update', audit_update)
    event.listen(cls, 'after_delete', audit_delete)