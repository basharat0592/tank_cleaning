import os
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI
import pandas as pd
import json
import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import pdfplumber
import time
import uuid
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Tank Cleaning Graph Analyzer",
    page_icon="🛢️",
    layout="wide"
)

# Initialize session state
if 'file_type' not in st.session_state:
    st.session_state.file_type = "CSV"
if 'table_select' not in st.session_state:
    st.session_state.table_select = "Documents"
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT_MODEL = os.getenv("AZURE_DEPLOYMENT_MODEL")

# PostgreSQL settings
POSTGRES_CONN = {
    "host": os.getenv("POSTGRES_HOST", "postgres_age_pgvector_prod"),
    "port": "5432",
    "dbname": "postgres",
    "user": "postgres",
    "password": "mysecretpassword",
}
GRAPH_NAME = "tank_cleaning_graph"

# Initialize Azure OpenAI client
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY, 
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,  
    )

# Initialize SentenceTransformer for embeddings
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Google Translator
@st.cache_resource
def get_translator():
    return GoogleTranslator(source='nl', target='en')

client = get_openai_client()
embedding_model = get_embedding_model()
translator = get_translator()

def call_azure_openai(messages):
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_MODEL,
            messages=messages,
            max_tokens=500
        )    
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling Azure OpenAI: {e}")
        st.error(f"Error calling Azure OpenAI: {e}")
        return None

def check_and_create_documents_table():
    max_retries = 3
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**POSTGRES_CONN)
            conn.autocommit = True
            cur = conn.cursor()
            # Ensure vector extension is created
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Vector extension ensured.")
            # Check if documents table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                );
            """)
            table_exists = cur.fetchone()[0]
            if not table_exists:
                logger.info("Documents table does not exist. Creating...")
                cur.execute("""
                    CREATE TABLE documents (
                        id SERIAL PRIMARY KEY,
                        content_dutch TEXT,
                        content_english TEXT,
                        embedding VECTOR(384),
                        source TEXT,
                        created_at TIMESTAMP
                    );
                """)
                logger.info("Documents table created successfully.")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to check/create documents table: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()
    logger.error("Failed to check/create documents table after all retries.")
    st.error("Failed to ensure documents table exists. Check database connection and logs.")
    return False

def initialize_database():
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        if not check_and_create_documents_table():
            logger.error("Failed to ensure documents table exists")
            return False
        cur.execute("SELECT * FROM ag_catalog.ag_graph WHERE name = %s;", (GRAPH_NAME,))
        if not cur.fetchone():
            cur.execute("SELECT ag_catalog.create_graph(%s);", (GRAPH_NAME,))
            logger.info(f"Created graph: {GRAPH_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.error(f"Error initializing database: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def store_document(content_dutch, source, file_type):
    try:
        if not check_and_create_documents_table():
            logger.error("Cannot store document: documents table unavailable")
            st.error("Cannot store document: database table unavailable")
            return None
        # Translate Dutch to English
        content_english = translator.translate(content_dutch) if content_dutch else ""
        if not content_english:
            logger.warning("Translation to English failed; storing empty English content")
            content_english = ""
        # Generate embedding from English content
        embedding = embedding_model.encode(content_english if content_english else " ").tolist()
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO documents (content_dutch, content_english, embedding, source, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (content_dutch, content_english, embedding_str, source, datetime.now()))
        doc_id = cur.fetchone()[0]
        entities_prompt = f"""
        Extract entities and relationships from this text for a tank cleaning context:
        Text: "{content_english}"
        Return as JSON with:
        - entities: list of {{type: 'Container|Issue|Person|Company|Report', value: string, id: string (if applicable)}}
        - relationships: list of {{source_type: string, source_id: string, target_type: string, target_id: string, type: string}}
        """
        entities_response = call_azure_openai([{"role": "user", "content": entities_prompt}])
        if entities_response:
            try:
                entities_data = json.loads(entities_response)
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                for entity in entities_data.get('entities', []):
                    label = entity['type']
                    props = entity.get('value', '')
                    entity_id = entity.get('id', f"{label}_{str(hash(props))[:8]}")
                    escaped_props = props.replace("'", "''")
                    cypher = f"CREATE (:{label} {{id: '{entity_id}', name: '{escaped_props}'}})"
                    cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
                for rel in entities_data.get('relationships', []):
                    cypher = (
                        f"MATCH (s:{rel['source_type']} {{id: '{rel['source_id']}'}}), "
                        f"(t:{rel['target_type']} {{id: '{rel['target_id']}'}}) "
                        f"CREATE (s)-[:{rel['type']}]->(t)"
                    )
                    cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse entities response: {e}")
        return doc_id
    except Exception as e:
        logger.error(f"Error storing document: {e}")
        st.error(f"Error storing document: {e}")
        return None
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def semantic_search(query, limit=3):
    try:
        if not check_and_create_documents_table():
            logger.warning("Documents table does not exist for semantic search")
            return []
        query_embedding = embedding_model.encode(query).tolist()
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            SELECT id, content_dutch, content_english, source, created_at
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (embedding_str, limit))
        results = cur.fetchall()
        return [{
            "id": r[0],
            "content_dutch": r[1],
            "content_english": r[2],
            "source": r[3],
            "created_at": r[4].isoformat() if r[4] else None
        } for r in results]
    except Exception as e:
        logger.warning(f"Error in semantic search: {e}")
        st.warning(f"Error in semantic search: {e}")
        return []
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def generate_sample_csv(table_name):
    if table_name == "Customers":
        df = pd.DataFrame([
            {"Customer": "ChemGlobal Inc.", "ID": 1, "Address": "123 Industrial Rd", "Postalcode": "90210", "City": "Houston", "Country": "USA"},
            {"Customer": "BulkFood Logistics", "ID": 2, "Address": "25 Market Lane", "Postalcode": "30301", "City": "Atlanta", "Country": "USA"},
            {"Customer": "LiquidChem Solutions", "ID": 3, "Address": "456 Chem Ave", "Postalcode": "40202", "City": "Louisville", "Country": "USA"},
            {"Customer": "AquaFoods Ltd.", "ID": 4, "Address": "789 Harbor St", "Postalcode": "10001", "City": "New York", "Country": "USA"},
            {"Customer": "TankOps Enterprises", "ID": 5, "Address": "101 Tech Rd", "Postalcode": "98101", "City": "Seattle", "Country": "USA"},
            {"Customer": "FoodChem Transport", "ID": 6, "Address": "202 Food Dr", "Postalcode": "60601", "City": "Chicago", "Country": "USA"},
            {"Customer": "Chemistry Direct", "ID": 7, "Address": "303 Chem Blvd", "Postalcode": "77001", "City": "Houston", "Country": "USA"},
            {"Customer": "EcoBulk Solutions", "ID": 8, "Address": "404 Green St", "Postalcode": "94101", "City": "San Francisco", "Country": "USA"},
            {"Customer": "GlobalChem Partners", "ID": 9, "Address": "505 Industry Way", "Postalcode": "90001", "City": "Los Angeles", "Country": "USA"},
            {"Customer": "HydroFood Inc.", "ID": 10, "Address": "606 Water St", "Postalcode": "70112", "City": "New Orleans", "Country": "USA"},
            {"Customer": "ChemBulk Logistics", "ID": 11, "Address": "707 Bulk Rd", "Postalcode": "28201", "City": "Charlotte", "Country": "USA"},
        ])
    elif table_name == "EFTCO_Codes":
        df = pd.DataFrame([
            {"Code": "C", "ID": 1, "Cleaning agent": "Cleaning agents", "Guideline": ""},
            {"Code": "C01", "ID": 2, "Cleaning agent": "Alkaline detergent", "Guideline": "Detergent with pH >7..."},
        ])
    elif table_name == "Fleetserie":
        df = pd.DataFrame([
            {"Fleetserie number": 54001, "FleetserieID": 1, "Manufacturer": "Singamas", "Inner tank material": "Composite", "Insulation": False, "Insulation type": ""},
            {"Fleetserie number": 54002, "FleetserieID": 2, "Manufacturer": "CIMC", "Inner tank material": "Stainless steel", "Insulation": True, "Insulation type": "Glass wool"},
        ])
    elif table_name == "Products":
        df = pd.DataFrame([
            {"Product": "Water", "ID": 1, "Manufacturer": "Universal Solvents Inc.", "Common name": "Water", "PH value": 7.0, "Water solutability": 0.0035, "Viscosity (mPas.s)": 0.89},
            {"Product": "Ethyl alcohol", "ID": 2, "Manufacturer": "ChemCo Ltd.", "Common name": "Ethanol", "PH value": 7.2, "Water solutability": 0.872, "Viscosity (mPas.s)": 1.2},
        ])
    elif table_name == "TankContainers":
        df = pd.DataFrame([{
            "Tank number": "HOYU 000001-3",
            "TankID": 1,
            "Fleetserie number": 54004,
            "Operator": "Hoyer",
            "Type": "T50",
            "Manlid holes": 2,
            "Capacity (L)": 22500,
            "Baffles": False,
            "Coating": "None",
            "Previous Product 1": "Specflex",
            "Previous Product 2": "Methanol",
            "Previous Product 3": "Methanol"
        }])
    elif table_name == "Documents":
        df = pd.DataFrame([
            {
                "Content_Dutch": "Container HOYU 000001-3 had een kleplekkage probleem op 3 mei, gerapporteerd door Chauffeur John.",
                "Content_English": "Container HOYU 000001-3 had a valve leakage issue on May 3, reported by Driver John.",
                "Source": "Driver Report",
                "Created_at": "2025-05-03 10:00:00"
            },
        ])
    else:
        df = pd.DataFrame()
    return df.to_csv(index=False)

def check_node_label(label):
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        query = f"MATCH (n:{label}) RETURN n LIMIT 1"
        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${query}$$) as (n agtype);")
        return bool(cur.fetchone())
    except Exception as e:
        logger.warning(f"Error checking node label {label}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def create_node_label(label):
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        if label == "Customer":
            cypher = "CREATE (:Customer {id: -1, name: 'DUMMY', address: '', postalcode: '', city: '', country: ''})"
        elif label == "EFTCOCode":
            cypher = "CREATE (:EFTCOCode {id: -1, code: 'DUMMY', agent: '', guideline: ''})"
        elif label == "Fleetserie":
            cypher = "CREATE (:Fleetserie {id: -1, number: 'DUMMY', manufacturer: '', material: '', insulation: false, insulation_type: ''})"
        elif label == "Product":
            cypher = "CREATE (:Product {id: -1, name: 'DUMMY', manufacturer: '', common_name: '', pH: 0.0, solubility: 0.0, viscosity: 0.0})"
        elif label == "TankContainer":
            cypher = "CREATE (:TankContainer {id: -1, number: 'DUMMY', fleetserie: '', operator: '', type: '', manlid_holes: 0, capacity_l: 0, baffles: false, coating: '', prev1: '', prev2: '', prev3: ''})"
        elif label in ["Container", "Issue", "Person", "Company", "Report"]:
            cypher = f"CREATE (:{label} {{id: '-1', name: 'DUMMY'}})"
        else:
            return False
        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
        logger.info(f"Created node label: {label}")
        return True
    except Exception as e:
        logger.error(f"Error creating node label {label}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def delete_existing_data(table_name):
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        node_labels = {
            "Customers": "Customer",
            "EFTCO_Codes": "EFTCOCode",
            "Fleetserie": "Fleetserie",
            "Products": "Product",
            "TankContainers": "TankContainer"
        }
        if table_name not in node_labels:
            return False
        cypher = f"MATCH (n:{node_labels[table_name]}) DETACH DELETE n"
        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
        return True
    except Exception as e:
        logger.error(f"Error deleting existing data for {table_name}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def import_csv_to_graph(table_name, uploaded_file, file_type="csv"):
    try:
        if not initialize_database():
            logger.error("Failed to initialize database")
            st.error("Failed to initialize database")
            return False
        node_labels = {
            "Customers": "Customer",
            "EFTCO_Codes": "EFTCOCode",
            "Fleetserie": "Fleetserie",
            "Products": "Product",
            "TankContainers": "TankContainer",
            "Documents": "Document"
        }
        if table_name not in node_labels:
            logger.error(f"Invalid table name: {table_name}")
            st.error(f"Invalid table name: {table_name}")
            return False
        label = node_labels[table_name]
        if table_name != "Documents":
            if not check_node_label(label):
                if not create_node_label(label):
                    logger.error(f"Failed to create node label {label}")
                    st.error(f"Failed to create node label {label}")
                    return False
                logger.info(f"Initialized node label {label}")
            else:
                if not delete_existing_data(table_name):
                    logger.error(f"Failed to delete existing {table_name} data")
                    st.error(f"Failed to delete existing {table_name} data")
                    return False
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        if table_name == "Documents":
            if file_type == "csv":
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', keep_default_na=False)
                for _, row in df.iterrows():
                    content_dutch = row.get('Content_Dutch', row.get('Content', '')).replace("'", "''")
                    content_english = row.get('Content_English', '').replace("'", "''")
                    source = row.get('Source', 'Unknown').replace("'", "''")
                    if not content_english and content_dutch:
                        content_english = translator.translate(content_dutch) or ""
                    if not store_document(content_dutch, source, file_type):
                        logger.error("Failed to store document from CSV")
                        st.error("Failed to store document from CSV")
                        return False
            elif file_type == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if not text.strip():
                        logger.error("No text extracted from PDF")
                        st.error("No text extracted from PDF. Ensure the PDF contains readable text.")
                        return False
                    content_dutch = text.replace("'", "''")
                    source = uploaded_file.name
                    if not store_document(content_dutch, source, file_type):
                        logger.error("Failed to store document from PDF")
                        st.error("Failed to store document from PDF")
                        return False
            logger.info(f"Documents imported successfully from {file_type}")
            st.success(f"Documents imported successfully from {file_type}")
        else:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', keep_default_na=False)
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            cypher_statements = []
            if table_name == "Customers":
                for _, row in df.iterrows():
                    customer_name = row['Customer'].replace("'", "''")
                    address = row['Address'].replace("'", "''")
                    city = row['City'].replace("'", "''")
                    country = row['Country'].replace("'", "''")
                    cypher = (
                        f"CREATE (:Customer {{id: {row['ID']}, "
                        f"name: '{customer_name}', "
                        f"address: '{address}', "
                        f"postalcode: '{row['Postalcode']}', "
                        f"city: '{city}', "
                        f"country: '{country}'}})"
                    )
                    cypher_statements.append(cypher)
            elif table_name == "EFTCO_Codes":
                for _, row in df.iterrows():
                    code = str(row.get('Code', '')).replace("'", "''")
                    id_value = int(row.get('ID', 0))
                    agent = str(row.get('Cleaning agent', '')).replace("'", "''")
                    guideline = str(row.get('Guideline', '')).replace("'", "''")
                    cypher = (
                        f"CREATE (:EFTCOCode {{id: {id_value}, "
                        f"code: '{code}', "
                        f"agent: '{agent}', "
                        f"guideline: '{guideline}'}})"
                    )
                    cypher_statements.append(cypher)
            elif table_name == "Fleetserie":
                for _, row in df.iterrows():
                    insulation_type = str(row['Insulation type']).replace("'", "''") if row['Insulation type'] else ''
                    manufacturer = str(row['Manufacturer']).replace("'", "''")
                    material = str(row['Inner tank material']).replace("'", "''")
                    cypher = (
                        f"CREATE (:Fleetserie {{id: {row['FleetserieID']}, "
                        f"number: '{row['Fleetserie number']}', "
                        f"manufacturer: '{manufacturer}', "
                        f"material: '{material}', "
                        f"insulation: {str(row['Insulation']).lower()}, "
                        f"insulation_type: '{insulation_type}'}})"
                    )
                    cypher_statements.append(cypher)
            elif table_name == "Products":
                for _, row in df.iterrows():
                    product_name = row['Product'].replace("'", "''")
                    manufacturer = row['Manufacturer'].replace("'", "''")
                    common_name = row['Common name'].replace("'", "''")
                    cypher = (
                        f"CREATE (:Product {{id: {row['ID']}, "
                        f"name: '{product_name}', "
                        f"manufacturer: '{manufacturer}', "
                        f"common_name: '{common_name}', "
                        f"pH: {row['PH value']}, "
                        f"solubility: {row['Water solutability']}, "
                        f"viscosity: {row['Viscosity (mPas.s)']}}})"
                    )
                    cypher_statements.append(cypher)
            elif table_name == "TankContainers":
                if not check_node_label("Product"):
                    if not create_node_label("Product"):
                        logger.error("Failed to create Product node label")
                        st.error("Failed to create Product node label")
                        return False
                if not check_node_label("Fleetserie"):
                    if not create_node_label("Fleetserie"):
                        logger.error("Failed to create Fleetserie node label")
                        st.error("Failed to create Fleetserie node label")
                        return False
                for _, row in df.iterrows():
                    coating = str(row['Coating']).replace("'", "''") if row['Coating'] else 'None'
                    operator = row['Operator'].replace("'", "''")
                    prev1 = row['Previous Product 1'].replace("'", "''")
                    prev2 = row['Previous Product 2'].replace("'", "''")
                    prev3 = row['Previous Product 3'].replace("'", "''")
                    cypher = (
                        f"CREATE (:TankContainer {{id: {row['TankID']}, "
                        f"number: '{row['Tank number']}', "
                        f"fleetserie: '{row['Fleetserie number']}', "
                        f"operator: '{operator}', "
                        f"type: '{row['Type']}', "
                        f"manlid_holes: {row['Manlid holes']}, "
                        f"capacity_l: {row['Capacity (L)']}, "
                        f"baffles: {str(row['Baffles']).lower()}, "
                        f"coating: '{coating}', "
                        f"prev1: '{prev1}', "
                        f"prev2: '{prev2}', "
                        f"prev3: '{prev3}'}})"
                    )
                    cypher_statements.append(cypher)
                    for idx, prev_field in enumerate(['Previous Product 1', 'Previous Product 2', 'Previous Product 3'], 1):
                        product_name = str(row[prev_field]).replace("'", "''")
                        product_exists = False
                        try:
                            product_df = pd.read_csv(generate_sample_csv("Products"), encoding='ISO-8859-1')
                            product_exists = product_name in product_df['Product'].values or product_name in product_df['Common name'].values
                        except Exception as e:
                            logger.warning(f"Error checking product existence: {e}")
                        if product_name and product_exists:
                            cypher = (
                                f"MATCH (t:TankContainer {{number: '{row['Tank number']}'}}), "
                                f"(p:Product) WHERE p.name = '{product_name}' OR p.common_name = '{product_name}' "
                                f"CREATE (t)-[:HAS_PREVIOUS_PRODUCT {{sequence: {idx}}}]->(p)"
                            )
                            cypher_statements.append(cypher)
                    fleetserie_number = str(row['Fleetserie number'])
                    if fleetserie_number:
                        cypher = (
                            f"MATCH (t:TankContainer {{number: '{row['Tank number']}'}}), "
                            f"(f:Fleetserie {{number: '{fleetserie_number}'}}) "
                            f"CREATE (t)-[:PART_OF]->(f)"
                        )
                        cypher_statements.append(cypher)
            for stmt in cypher_statements:
                try:
                    cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${stmt}$$) as (result agtype);")
                except Exception as e:
                    logger.error(f"Error executing statement: {stmt}\nError: {e}")
                    continue
            for label in node_labels.values():
                if label != "Document":
                    cypher = f"MATCH (n:{label} {{id: -1}}) DETACH DELETE n"
                    try:
                        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
                    except Exception as e:
                        logger.warning(f"Error deleting dummy node for {label}: {e}")
        logger.info(f"Data imported successfully into {table_name}")
        st.success(f"Data imported successfully into {table_name}")
        return True
    except Exception as e:
        logger.error(f"Error importing data for {table_name}: {e}")
        st.error(f"Error importing data for {table_name}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def validate_cypher_query(cypher_query: str) -> bool:
    allowed_terms = [
        "TANKCONTAINER", "CUSTOMER", "PRODUCT", "FLEETSERIE", "EFTCOCODE",
        "CONTAINER", "ISSUE", "PERSON", "COMPANY", "REPORT",
        "HAS_PREVIOUS_PRODUCT", "PART_OF", "BELONGS_TO", "HAS_ISSUE", "MENTIONS"
    ]
    query_upper = cypher_query.upper()
    return any(term in query_upper for term in allowed_terms) or "MATCH" in query_upper

def run_cypher_query(cypher_query: str):
    try:
        cypher_query = cypher_query.strip()
        cypher_query = re.sub(r'```cypher|```', '', cypher_query)
        cypher_query = ' '.join(cypher_query.split())
        cypher_query = cypher_query.rstrip(';')
        return_clause = cypher_query[cypher_query.upper().find('RETURN')+6:].strip()
        clean_return = re.sub(r'\s+AS\s+\w+', '', return_clause)
        return_items = [col.strip() for col in clean_return.split(',')]
        # Generate unique column names to avoid duplicates
        columns = []
        seen = {}
        for item in return_items:
            col_name = item.split('.')[-1]
            if col_name in seen:
                seen[col_name] += 1
                columns.append(f"{col_name}_{seen[col_name]}")
            else:
                seen[col_name] = 0
                columns.append(col_name)
        column_def = ', '.join(f"{col} agtype" for col in columns)
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        formatted_query = f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher_query}$$) as ({column_def});"
        cur.execute(formatted_query)
        rows = cur.fetchall()
        processed_results = []
        for row in rows:
            result_dict = {}
            for idx, col in enumerate(columns):
                agtype_str = str(row[idx])
                try:
                    if agtype_str.startswith('{') or agtype_str.startswith('['):
                        result_dict[col] = json.loads(agtype_str)
                    else:
                        result_dict[col] = agtype_str
                except json.JSONDecodeError:
                    result_dict[col] = agtype_str
            processed_results.append(result_dict)
        return processed_results
    except Exception as e:
        logger.warning(f"Error executing Cypher query: {str(e)}")
        st.warning(f"Error executing Cypher query: {str(e)}")
        return []
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

cypher_prompt = PromptTemplate(
    template="""Generate a valid Cypher query for Apache AGE graph database based on the provided schema and question.

SCHEMA:
- Nodes:
  - TankContainer(id: Integer, number: String, fleetserie: String, operator: String, type: String, manlid_holes: Integer, capacity_l: Integer, baffles: Boolean, coating: String, prev1: String, prev2: String, prev3: String)
  - Customer(id: Integer, name: String, address: String, postalcode: String, city: String, country: String)
  - Product(id: Integer, name: String, manufacturer: String, common_name: String, pH: Float, solubility: Float, viscosity: Float)
  - Fleetserie(id: Integer, number: String, manufacturer: String, material: String, insulation: Boolean, insulation_type: String)
  - EFTCOCode(id: Integer, code: String, agent: String, guideline: String)
  - Container(id: String, name: String)
  - Issue(id: String, name: String)
  - Person(id: String, name: String)
  - Company(id: String, name: String)
  - Report(id: String, name: String)
- Relationships:
  - (TankContainer)-[:HAS_PREVIOUS_PRODUCT]->(Product)
  - (TankContainer)-[:PART_OF]->(Fleetserie)
  - (TankContainer)-[:BELONGS_TO]->(Customer)
  - (Container)-[:HAS_ISSUE]->(Issue)
  - (Report)-[:MENTIONS]->(Container)
  - (Report)-[:REPORTED_BY]->(Person)
  - (Report)-[:CLEANED_BY]->(Company)
- Notes:
  - TankContainer has prev1, prev2, prev3 fields storing previous product names.
  - Use prev1, prev2, prev3 for queries about previous products.
  - Product names should match against both 'name' and 'common_name' fields.
  - BELONGS_TO relationships may not exist.
  - Not all previous products are in the Product table, so rely on prev1, prev2, prev3.
  - For fleet series queries, use PART_OF relationships or TankContainer.fleetserie.
  - For previous product queries, return prev1, prev2, prev3 directly.
  - Avoid AS aliases in RETURN clause for Apache AGE compatibility.
  - For issues or reports, use Container, Issue, Person, Company, and Report nodes.

RULES:
1. Use only defined nodes and relationships.
2. Return a single Cypher query as plain text (no markdown, no semicolons).
3. Include a RETURN clause with specific fields.
4. Use single quotes for string literals.
5. Ensure syntactic correctness for Apache AGE.
6. Handle missing relationships by focusing on available data.

Question: "{question}"
""",
    input_variables=["question"],
)

def process_user_query(user_question):
    MAX_ATTEMPTS = 3
    attempts = 0

    # Handle queries explicitly requesting document contents
    if "contents of documents" in user_question.lower():
        try:
            conn = psycopg2.connect(**POSTGRES_CONN)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SELECT id, content_dutch, content_english, source, created_at FROM public.documents ORDER BY id ASC")
            results = cur.fetchall()
            doc_results = [{
                "id": r[0],
                "content_dutch": r[1],
                "content_english": r[2],
                "source": r[3],
                "created_at": r[4].isoformat() if r[4] else None
            } for r in results]
            cur.close()
            conn.close()

            # Generate a human-readable response
            if doc_results:
                response = "Here are the contents of the documents:\n"
                source_info = "Source of Information:\n- Structured: Documents table\n- Details:\n"
                file_type = st.session_state.file_type.lower() if 'file_type' in st.session_state else "unknown"
                for doc in doc_results:
                    response += f"- Document ID {doc['id']} (Source: {doc['source']}, Created: {doc['created_at']}):\n"
                    response += f"  Dutch: {doc['content_dutch']}\n"
                    response += f"  English: {doc['content_english']}\n"
                    source_info += f"  - Document ID {doc['id']}: Originally from {file_type.upper()} file ('{doc['source']}'), stored in Dutch and English\n"
                response += "\n" + source_info
            else:
                response = "No documents found in the database.\n\nSource of Information: No data retrieved from documents table."
            return None, doc_results, response
        except Exception as e:
            logger.error(f"Error fetching document contents: {e}")
            st.error(f"Error fetching document contents: {e}")
            return None, [], "Error retrieving document contents."

    # Existing logic for other queries
    while attempts < MAX_ATTEMPTS:
        doc_results = semantic_search(user_question)
        doc_context = "\n".join([f"Document {d['id']} (Source: {d['source']}): {d['content_english']}" for d in doc_results]) if doc_results else "No relevant documents found."
        cypher_query = call_azure_openai([{
            "role": "user", 
            "content": cypher_prompt.format(question=user_question + f"\nRelevant documents:\n{doc_context}")
        }])
        if not cypher_query:
            attempts += 1
            continue
        if not validate_cypher_query(cypher_query):
            attempts += 1
            logger.warning(f"Invalid query, attempt {attempts}/{MAX_ATTEMPTS}: {cypher_query}")
            continue
        result = run_cypher_query(cypher_query)
        combined_context = {
            "graph_results": result,
            "document_results": doc_results
        }
        response = call_azure_openai([{
            "role": "user",
            "content": f"""
            Explain these tank cleaning results in simple terms: {json.dumps(combined_context, default=str)}.
            The user asked: '{user_question}'.
            Keep the response concise and focused on the key findings.
            For previous product queries, refer to the fields as Previous Product 1, Previous Product 2, and Previous Product 3.
            If no documents were found, note that no relevant documents were available.
            """
        }])
        if response:
            # Add source of information
            source_info = "Source of Information:\n"
            if result:
                # Extract node labels from the Cypher query
                node_labels = re.findall(r'\((\w+):(\w+)', cypher_query)
                node_types = {label for _, label in node_labels}
                source_info += f"- Structured: Graph database (Nodes: {', '.join(node_types)})\n"
            if doc_results:
                file_type = st.session_state.file_type.lower() if 'file_type' in st.session_state else "unknown"
                source_info += "- Structured: Documents table (via semantic search, English embeddings)\n- Details:\n"
                for doc in doc_results:
                    source_info += f"  - Document ID {doc['id']}: Originally from {file_type.upper()} file ('{doc['source']}'), stored in Dutch and English\n"
            if not result and not doc_results:
                source_info += "- No data retrieved from graph or documents table."
            response += "\n\n" + source_info
            return cypher_query, result, response
        attempts += 1

    # Fallback logic
    fallback_cypher = None
    if "previous products" in user_question.lower():
        tank_number = re.search(r'HOYU \d+-\d', user_question)
        if tank_number:
            fallback_cypher = f"MATCH (t:TankContainer {{number: '{tank_number.group()}'}}) RETURN t.prev1, t.prev2, t.prev3"
    elif "capacity" in user_question.lower():
        capacity = re.search(r'(\d+)\s*liters', user_question)
        if capacity:
            fallback_cypher = f"MATCH (t:TankContainer) WHERE t.capacity_l > {capacity.group(1)} RETURN t.id, t.number, t.capacity_l, t.operator"
    elif "previously contained" in user_question.lower():
        product_name = re.search(r'contained\s+([^\?]+)', user_question, re.IGNORECASE)
        if product_name:
            product = product_name.group(1).strip().replace("'", "''")
            fallback_cypher = f"MATCH (t:TankContainer) WHERE t.prev1 = '{product}' OR t.prev2 = '{product}' OR t.prev3 = '{product}' RETURN t.id, t.number, t.operator"
    elif "customer" in user_question.lower() and "list" in user_question.lower():
        fallback_cypher = f"MATCH (c:Customer) RETURN c.id, c.name, c.address, c.postalcode, c.city, c.country"
    elif "issue" in user_question.lower() or "report" in user_question.lower():
        container_match = re.search(r'HOYU \d+-\d', user_question)
        if container_match:
            fallback_cypher = (
                f"MATCH (c:Container {{name: '{container_match.group()}'}})-[:HAS_ISSUE]->(i:Issue) "
                f"OPTIONAL MATCH (r:Report)-[:MENTIONS]->(c) "
                f"OPTIONAL MATCH (r)-[:REPORTED_BY]->(p:Person) "
                f"OPTIONAL MATCH (r)-[:CLEANED_BY]->(comp:Company) "
                f"RETURN c.name, i.name, p.name, comp.name"
            )
    if fallback_cypher:
        result = run_cypher_query(fallback_cypher)
        doc_results = semantic_search(user_question)
        response = call_azure_openai([{
            "role": "user",
            "content": f"""
            Explain these tank cleaning results in simple terms: {json.dumps({'graph_results': result, 'document_results': doc_results}, default=str)}.
            The user asked: '{user_question}'.
            Keep the response concise and focused on the key findings.
            For previous product queries, refer to the fields as Previous Product 1, Previous Product 2, and Previous Product 3.
            If no documents were found, note that no relevant documents were available.
            """
        }]) if result or doc_results else f"No results found for '{user_question}'. No relevant documents or graph data available."
        # Add source of information for fallback
        source_info = "Source of Information:\n"
        if result:
            node_labels = re.findall(r'\((\w+):(\w+)', fallback_cypher)
            node_types = {label for _, label in node_labels}
            source_info += f"- Structured: Graph database (Nodes: {', '.join(node_types)})\n"
        if doc_results:
            file_type = st.session_state.file_type.lower() if 'file_type' in st.session_state else "unknown"
            source_info += "- Structured: Documents table (via semantic search, English embeddings)\n- Details:\n"
            for doc in doc_results:
                source_info += f"  - Document ID {doc['id']}: Originally from {file_type.upper()} file ('{doc['source']}'), stored in Dutch and English\n"
        if not result and not doc_results:
            source_info += "- No data retrieved from graph or documents table."
        response += "\n\n" + source_info
        return fallback_cypher, result, response
    return None, None, f"No results found for '{user_question}'. No relevant documents or graph data available.\n\nSource of Information: No data retrieved from graph or documents table."

# Streamlit UI
st.title("🛢️ Tank Cleaning Graph Analyzer")
st.markdown("""
This application queries a tank cleaning graph database using natural language.
It combines Apache AGE for graph queries, PGVector for semantic search, and Azure OpenAI.
""")

# Sidebar for initialization
with st.sidebar:
    st.header("Sample Questions")
    st.markdown("""
    Try asking:
    - What are the previous products in tank HOYU 000001-3?
    - Which tanks have capacity over 20000 liters?
    - Which tanks previously contained Methanol?
    - What issues were reported for container HOYU 000001-3?
    - Give me contents of documents
    """)
    st.header("📥 Import Data")
    
    table_select = st.selectbox(
        "Select Table to Import",
        ["Customers", "EFTCO_Codes", "Fleetserie", "Products", "TankContainers", "Documents"],
        key="table_select"
    )
    
    if table_select != st.session_state.table_select:
        st.session_state.table_select = table_select
        st.session_state.file_type = "CSV"
        st.session_state.file_uploader_key += 1
    
    file_type = "csv"
    accept_types = [".csv"]
    if table_select == "Documents":
        file_type = st.radio(
            "File Type",
            ["CSV", "PDF"],
            index=1 if st.session_state.file_type == "PDF" else 0,
            key="file_type_radio",
            help="Select 'PDF' to upload PDF files or 'CSV' for CSV files."
        )
        if file_type != st.session_state.file_type:
            st.session_state.file_type = file_type
            st.session_state.file_uploader_key += 1
        accept_types = [".pdf"] if file_type == "PDF" else [".csv"]
    
    st.write(f"**Selected Table**: {table_select}<br>**Selected File Type**: {file_type}", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload File",
        type=accept_types,
        key=f"uploader_{st.session_state.file_uploader_key}",
        help=f"Upload a {file_type} file for {table_select}."
    )
    
    if uploaded_file:
        if table_select == "Documents" and file_type == "CSV" and uploaded_file.type == "application/pdf":
            st.error("Select 'PDF' file type to upload a PDF file.")
            uploaded_file = None
        elif table_select == "Documents" and file_type == "PDF" and uploaded_file.type != "application/pdf":
            st.error("Upload a PDF file when 'PDF' file type is selected.")
            uploaded_file = None
        elif table_select != "Documents" and uploaded_file.type != "text/csv":
            st.error(f"Upload a CSV file for {table_select}.")
            uploaded_file = None
    
    if uploaded_file and st.button("Import"):
        with st.spinner(f"Importing {table_select} data..."):
            success = import_csv_to_graph(table_select, uploaded_file, file_type.lower())
            if not success:
                st.error(f"Import failed for {table_select}. Check logs for details.")
    
    st.header("📤 Export Sample CSV")
    for table in ["Customers", "EFTCO_Codes", "Fleetserie", "Products", "TankContainers", "Documents"]:
        st.download_button(
            label=f"Download {table} Sample",
            data=generate_sample_csv(table),
            file_name=f"{table}_sample.csv",
            mime="text/csv"
        )

# Query interface
st.header("Ask a Question")
user_question = st.text_input(
    "Enter your question about tank cleaning:",
    placeholder="e.g., What issues were reported for container HOYU 000001-3?"
)
if user_question:
    with st.spinner("Processing your question..."):
        cypher, result, answer = process_user_query(user_question)
        if answer:
            st.subheader("Response")
            st.write(answer)
            # if cypher:
            #     st.subheader("Generated Cypher Query")
            #     st.code(cypher, language="cypher")
            # if result:
            #     st.subheader("Results")
            #     st.json(result)
        else:
            st.info("No results found.")

# Footer
st.markdown("---")
st.caption("Tank Cleaning Graph Analyzer | Powered by Apache AGE, PGVector, and Azure OpenAI")