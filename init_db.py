# docker compose up --build
# docker compose down -v

import time
import pyodbc
import re
import os
from flask import g
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from extensions import db
from app import app

SCHEMA_FILE = 'create_tables.txt'

DATA_FILES = [
    'create_biomarker_types.txt',
    'create_eeg.txt',
    'create_study_types.txt',
    'create_wearables.txt',
    'create_meddra.txt',
    'create_expense_categories.txt'
]

DB_SERVER = 'sql,1433'  # SQL Server service name and port
DB_NAME = 'novasyne'
DB_USER = 'sa'
DB_PASSWORD = 'YourStrongP@ssw0rd!' 

def wait_for_db():
    while True:
        try:

            conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DB_SERVER};"
                f"DATABASE=master;"
                f"UID={DB_USER};"
                f"PWD={DB_PASSWORD};",
                timeout=3
            )
            conn.close()
            print("Database server connection successful.")
            break
        except Exception as e:
            print(f"Waiting for SQL Server... Error: {str(e)}")
            time.sleep(3)


def create_database_if_not_exists():
    print(f"Checking if database '{DB_NAME}' exists...")
    
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE=master;"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};"
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT database_id FROM sys.databases WHERE name = ?",
            (DB_NAME,)
        )
        
        if cursor.fetchone() is not None:
            print(f"Database '{DB_NAME}' already exists. Skipping initialization.")
            cursor.close()
            conn.close()
            return False  # Database exists, don't initialize
        
        print(f"Database '{DB_NAME}' does not exist. Creating it...")
        
        create_db_sql = f"CREATE DATABASE [{DB_NAME}] COLLATE SQL_Latin1_General_CP1_CI_AS;"
        cursor.execute(create_db_sql)
        print(f"Database '{DB_NAME}' created successfully.")
        cursor.close()
        conn.close()
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE=master;"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Configuring database settings...")
        settings = [
            f"ALTER DATABASE [{DB_NAME}] SET COMPATIBILITY_LEVEL = 160",
            f"ALTER DATABASE [{DB_NAME}] SET ANSI_NULL_DEFAULT OFF",
            f"ALTER DATABASE [{DB_NAME}] SET ANSI_NULLS OFF",
            f"ALTER DATABASE [{DB_NAME}] SET ANSI_PADDING OFF",
            f"ALTER DATABASE [{DB_NAME}] SET ANSI_WARNINGS OFF",
            f"ALTER DATABASE [{DB_NAME}] SET ARITHABORT OFF",
            f"ALTER DATABASE [{DB_NAME}] SET AUTO_SHRINK OFF",
            f"ALTER DATABASE [{DB_NAME}] SET AUTO_UPDATE_STATISTICS ON",
            f"ALTER DATABASE [{DB_NAME}] SET CURSOR_CLOSE_ON_COMMIT OFF",
            f"ALTER DATABASE [{DB_NAME}] SET CONCAT_NULL_YIELDS_NULL OFF",
            f"ALTER DATABASE [{DB_NAME}] SET NUMERIC_ROUNDABORT OFF",
            f"ALTER DATABASE [{DB_NAME}] SET QUOTED_IDENTIFIER OFF",
            f"ALTER DATABASE [{DB_NAME}] SET RECURSIVE_TRIGGERS OFF",
            f"ALTER DATABASE [{DB_NAME}] SET AUTO_UPDATE_STATISTICS_ASYNC OFF",
            f"ALTER DATABASE [{DB_NAME}] SET ALLOW_SNAPSHOT_ISOLATION ON",
            f"ALTER DATABASE [{DB_NAME}] SET PARAMETERIZATION SIMPLE",
            f"ALTER DATABASE [{DB_NAME}] SET READ_COMMITTED_SNAPSHOT ON",
            f"ALTER DATABASE [{DB_NAME}] SET MULTI_USER",
        ]
        
        for setting in settings:
            try:
                cursor.execute(setting)
            except Exception as e:
                print(f"Warning: Could not apply setting: {setting}")
                print(f"  Error: {e}")
        
        try:
            cursor.execute(f"ALTER DATABASE [{DB_NAME}] SET QUERY_STORE = ON")
            cursor.execute(f"""
                ALTER DATABASE [{DB_NAME}] SET QUERY_STORE (
                    OPERATION_MODE = READ_WRITE,
                    CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30),
                    DATA_FLUSH_INTERVAL_SECONDS = 900,
                    INTERVAL_LENGTH_MINUTES = 60,
                    MAX_STORAGE_SIZE_MB = 100,
                    QUERY_CAPTURE_MODE = AUTO,
                    SIZE_BASED_CLEANUP_MODE = AUTO,
                    MAX_PLANS_PER_QUERY = 200,
                    WAIT_STATS_CAPTURE_MODE = ON
                )
            """)
            print("Query Store configured successfully.")
        except Exception as e:
            print(f"Query Store configuration skipped: {e}")
        
        try:
            cursor.execute(f"ALTER DATABASE [{DB_NAME}] SET READ_WRITE")
        except Exception as e:
            print(f"Warning: Could not set READ_WRITE mode: {e}")
            
        print("Database settings configured.")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error in create_database_if_not_exists: {e}")
        raise


def execute_sql_file(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found, skipping: {filepath}")
        return

    print(f"Executing script: {filepath}...")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    batches = re.split(r'^\s*GO\s*$', content, flags=re.IGNORECASE | re.MULTILINE)

    executed_batches = 0
    try:
        for batch in batches:
            batch = batch.strip()

            if not batch:
                continue
                
            lines = [line.strip() for line in batch.splitlines() if line.strip()]
            if not lines:
                continue
                
            is_comment_only = all(
                line.startswith('--') or line.startswith('/*')
                for line in lines
            )
            if is_comment_only:
                continue

            if batch.upper().strip().startswith('USE '):
                print(f"Skipping USE statement (already connected to {DB_NAME})")
                continue

            db.session.execute(text(batch))
            executed_batches += 1

        db.session.commit()
        print(f"Successfully executed {executed_batches} batches from {filepath}.")

    except SQLAlchemyError as e:
        print(f"\n--- ERROR EXECUTING BATCH IN {filepath} ---")
        print(f"Failed Batch Preview: {batch[:250]}...")
        print(f"Error: {e.orig}")
        db.session.rollback()
        raise
    except Exception as e:
        print(f"An unexpected error occurred with {filepath}: {e}")
        db.session.rollback()
        raise


def create_schema_and_seed():
    wait_for_db()
    needs_initialization = create_database_if_not_exists()
    
    if not needs_initialization:
        print("\n" + "="*60)
        print("DATABASE ALREADY EXISTS - SKIPPING ALL INITIALIZATION")
        print("="*60)
        print("If you want to reinitialize, use: docker-compose down -v")
        return 
    
    print("\n" + "="*60)
    print("NEW DATABASE CREATED - PROCEEDING WITH INITIALIZATION")
    print("="*60)
    time.sleep(2)

    print(f"Verifying connection to new database '{DB_NAME}'...")
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_NAME};"
            f"UID={DB_USER};"
            f"PWD={DB_PASSWORD};",
            timeout=5
        )
        conn.close()
        print(f"Successfully connected to fresh database '{DB_NAME}'.")
    except Exception as e:
        print(f"ERROR: Cannot connect to new database: {e}")
        raise

    with app.app_context():
        print("Disposing of old database connections...")
        db.engine.dispose()

        g.disable_auditing = True
        
        try:
            print("Creating database schema...")
            execute_sql_file(SCHEMA_FILE)
            print("Schema created successfully.")
            print("Seeding data...")
            for data_file in DATA_FILES:
                execute_sql_file(data_file)
            print("Data seeding complete.")

        except Exception as e:
            print("\nDatabase initialization FAILED.")
            
        finally:
            g.disable_auditing = False


if __name__ == "__main__":
    print("\nStarting Database Initialization...")
    create_schema_and_seed()
    print("Database Initialization Script Finished.")