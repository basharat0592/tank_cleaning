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
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT_MODEL = os.getenv("AZURE_DEPLOYMENT_MODEL")

# PostgreSQL settings
POSTGRES_CONN = {
    "host": "postgres",
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

client = get_openai_client()

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

# Export sample CSVs
def generate_sample_csv(table_name):
    if table_name == "Customers":
        df = pd.DataFrame([
            {"Customer": "ChemGlobal Inc.", "ID": 1, "Address": "123 Industrial Rd", "Postalcode": "90210", "City": "Houston", "Country": "USA"},
            {"Customer": "BulkFood Logistics", "ID": 2, "Address": "25 Market Lane", "Postalcode": "30301", "City": "Atlanta", "Country": "USA"},
            {"Customer": "LiquidChem Solutions", "ID": 3, "Address": "78 Chemical Ave", "Postalcode": "40001", "City": "Louisville", "Country": "USA"},
            {"Customer": "AquaFoods Ltd.", "ID": 4, "Address": "95 Ocean Blvd", "Postalcode": "11001", "City": "New York", "Country": "USA"},
            {"Customer": "TankOps Enterprises", "ID": 5, "Address": "67 Main St", "Postalcode": "98052", "City": "Seattle", "Country": "USA"},
            {"Customer": "FoodChem Transport", "ID": 6, "Address": "321 Pine Rd", "Postalcode": "60601", "City": "Chicago", "Country": "USA"},
            {"Customer": "Chemistry Direct", "ID": 7, "Address": "59 Laboratory Way", "Postalcode": "77001", "City": "Houston", "Country": "USA"},
            {"Customer": "EcoBulk Solutions", "ID": 8, "Address": "11 Green Rd", "Postalcode": "94101", "City": "San Francisco", "Country": "USA"},
            {"Customer": "GlobalChem Partners", "ID": 9, "Address": "200 International Blvd", "Postalcode": "90001", "City": "Los Angeles", "Country": "USA"},
            {"Customer": "HydroFood Inc.", "ID": 10, "Address": "88 Riverbank Rd", "Postalcode": "70112", "City": "New Orleans", "Country": "USA"},
            {"Customer": "ChemBulk Logistics", "ID": 11, "Address": "156 Chemical St", "Postalcode": "28202", "City": "Charlotte", "Country": "USA"},
            {"Customer": "LiquidLife Partners", "ID": 12, "Address": "43 Wellness Rd", "Postalcode": "75201", "City": "Dallas", "Country": "USA"},
            {"Customer": "PrimeFoodChem", "ID": 13, "Address": "72 Prime St", "Postalcode": "48201", "City": "Detroit", "Country": "USA"},
            {"Customer": "BioChemicals Ltd.", "ID": 14, "Address": "90 Bio St", "Postalcode": "30303", "City": "Atlanta", "Country": "USA"}
        ])
    elif table_name == "EFTCO_Codes":
        df = pd.DataFrame([
            {"Code": "C", "ID": 1, "Cleaning agent": "Cleaning agents", "Guideline": ""},
            {"Code": "C01", "ID": 2, "Cleaning agent": "Alkaline detergent", "Guideline": "Detergent with pH >7 used during the cleaning procedure..."}
        ])
    elif table_name == "Fleetserie":
        df = pd.DataFrame([
            {"Fleetserie number": 54001, "FleetserieID": 1, "Manufacturer": "Singamas", "Inner tank material": "Composite", "Insulation": False, "Insulation type": ""},
            {"Fleetserie number": 54002, "FleetserieID": 2, "Manufacturer": "CIMC", "Inner tank material": "Stainless steel", "Insulation": True, "Insulation type": "Glass wool"},
            {"Fleetserie number": 54003, "FleetserieID": 3, "Manufacturer": "Singamas", "Inner tank material": "Stainless steel", "Insulation": True, "Insulation type": "Glass wool"},
            {"Fleetserie number": 54004, "FleetserieID": 4, "Manufacturer": "UBH", "Inner tank material": "Stainless steel", "Insulation": False, "Insulation type": ""},
            {"Fleetserie number": 54005, "FleetserieID": 5, "Manufacturer": "Singamas", "Inner tank material": "Stainless steel", "Insulation": True, "Insulation type": "Glass wool"}
        ])
    elif table_name == "Products":
        df = pd.DataFrame([
            {"Product": "Water", "ID": 1, "Manufacturer": "Universal Solvents Inc.", "Common name": "Water", "PH value": 7.0, "Water solutability": 0.0035, "Viscosity (mPas.s)": 0.89},
            {"Product": "Ethyl alcohol, Grain alcohol", "ID": 2, "Manufacturer": "ChemCo Ltd.", "Common name": "Ethanol", "PH value": 7.2, "Water solutability": 0.872, "Viscosity (mPas.s)": 1.2},
            {"Product": "Methyl alcohol, Wood alcohol", "ID": 3, "Manufacturer": "Global Labs", "Common name": "Methanol", "PH value": 7.1, "Water solutability": 0.371, "Viscosity (mPas.s)": 0.59}
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
    else:
        df = pd.DataFrame()
    return df.to_csv(index=False)

def initialize_graph():
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        cur.execute("SELECT * FROM ag_catalog.ag_graph WHERE name = %s;", (GRAPH_NAME,))
        if not cur.fetchone():
            cur.execute("SELECT ag_catalog.create_graph(%s);", (GRAPH_NAME,))
            logger.info(f"Created graph: {GRAPH_NAME}")
            st.info(f"Created graph: {GRAPH_NAME}")
        return True
    except Exception as e:
        logger.error(f"Error initializing graph: {e}")
        st.error(f"Error initializing graph: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

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
        st.warning(f"Error checking node label {label}: {e}")
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
        # Create a dummy node to initialize the label
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
        else:
            return False
        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
        logger.info(f"Created node label: {label}")
        return True
    except Exception as e:
        logger.error(f"Error creating node label {label}: {e}")
        st.error(f"Error creating node label {label}: {e}")
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
        st.error(f"Error deleting existing data for {table_name}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def import_csv_to_graph(table_name, uploaded_file):
    try:
        if not initialize_graph():
            logger.error("Failed to initialize graph")
            st.error("Failed to initialize graph")
            return
        node_labels = {
            "Customers": "Customer",
            "EFTCO_Codes": "EFTCOCode",
            "Fleetserie": "Fleetserie",
            "Products": "Product",
            "TankContainers": "TankContainer"
        }
        if table_name not in node_labels:
            logger.error(f"Invalid table name: {table_name}")
            st.error(f"Invalid table name: {table_name}")
            return
        label = node_labels[table_name]
        # Check and create node label if it doesn't exist
        if not check_node_label(label):
            if not create_node_label(label):
                logger.error(f"Failed to create node label {label}")
                st.error(f"Failed to create node label {label}")
                return
            logger.info(f"Initialized node label {label}")
            st.info(f"Initialized node label {label}")
        else:
            # Delete existing data if label exists
            if not delete_existing_data(table_name):
                logger.error(f"Failed to delete existing {table_name} data")
                st.error(f"Failed to delete existing {table_name} data")
                return
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', keep_default_na=False)
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
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
            # Ensure dependent labels (Product, Fleetserie) exist
            if not check_node_label("Product"):
                if not create_node_label("Product"):
                    logger.error("Failed to create Product node label")
                    st.error("Failed to create Product node label")
                    return
                logger.info("Initialized Product node label")
                st.info("Initialized Product node label")
            if not check_node_label("Fleetserie"):
                if not create_node_label("Fleetserie"):
                    logger.error("Failed to create Fleetserie node label")
                    st.error("Failed to create Fleetserie node label")
                    return
                logger.info("Initialized Fleetserie node label")
                st.info("Initialized Fleetserie node label")
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
        # Delete dummy nodes
        for label in node_labels.values():
            cypher = f"MATCH (n:{label} {{id: -1}}) DETACH DELETE n"
            try:
                cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
            except Exception as e:
                logger.warning(f"Error deleting dummy node for {label}: {e}")
        # Execute import statements
        for stmt in cypher_statements:
            try:
                cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${stmt}$$) as (result agtype);")
            except Exception as e:
                logger.error(f"Error executing statement: {stmt}\nError: {e}")
                st.error(f"Error executing statement: {stmt}\nError: {e}")
                continue
        logger.info(f"Data imported successfully into {table_name}")
        st.success(f"Data imported successfully into {table_name}")
    except Exception as e:
        logger.error(f"Error importing data for {table_name}: {e}")
        st.error(f"Error importing data for {table_name}: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def validate_cypher_query(cypher_query: str) -> bool:
    allowed_terms = [
        "TANKCONTAINER", "CUSTOMER", "PRODUCT", "FLEETSERIE", "EFTCOCODE",
        "HAS_PREVIOUS_PRODUCT", "PART_OF", "BELONGS_TO"
    ]
    query_upper = cypher_query.upper()
    return any(term in query_upper for term in allowed_terms) or "MATCH" in query_upper

def show_graph_status(cypher_query: str):
    try:
        # Clean up the query - remove markdown code blocks and any extra whitespace
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        # Remove all line breaks and extra spaces to ensure single-line query
        cypher_query = ' '.join(cypher_query.split())
        
        # Replace any problematic characters in the query
        cypher_query = cypher_query.replace("'", "''")  # Escape single quotes
        
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        
        # Verify AGE extension is loaded
        cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        
        # Format the query properly for AGE - ensure it's a single line
        formatted_query = f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher_query}$$) as (result agtype);"
        
        # Debug output
        st.write("Final query being executed:")
        st.code(formatted_query, language="sql")
        
        cur.execute(formatted_query)
        rows = cur.fetchall()
        
        # Convert agtype results to Python objects
        processed_results = []
        for row in rows:
            if isinstance(row[0], dict):
                processed_results.append(row[0])
            else:
                processed_results.append(str(row[0]))
        
        return processed_results
    except Exception as e:
        st.error(f"Error executing Cypher query: {str(e)}")
        st.error("Please check the query syntax and try again.")
        return []
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def run_cypher_query(cypher_query: str):
    try:
        cypher_query = cypher_query.strip()
        cypher_query = re.sub(r'```cypher|```', '', cypher_query)
        cypher_query = ' '.join(cypher_query.split())
        cypher_query = cypher_query.rstrip(';')
        cypher_query = re.sub(r"''([^']*)''", r"'\1'", cypher_query)
        return_clause = cypher_query[cypher_query.upper().find('RETURN')+6:].strip()
        clean_return = re.sub(r'\s+AS\s+\w+', '', return_clause)
        columns = [col.strip().split('.')[-1] for col in clean_return.split(',')]
        column_def = ', '.join(f"{col} agtype" for col in columns)
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        formatted_query = f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher_query}$$) as ({column_def});"
        #logger.info(f"Executing Cypher query: {cypher_query}")
        #st.write("Executing Cypher query:")
        #st.code(cypher_query, language="cypher")
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
        st.code(cypher_query, language="cypher")
        clean_query = re.sub(r'\s+AS\s+\w+', '', cypher_query)
        try:
            conn = psycopg2.connect(**POSTGRES_CONN)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            return_clause = clean_query[clean_query.upper().find('RETURN')+6:].strip()
            columns = [col.strip().split('.')[-1] for col in return_clause.split(',')]
            column_def = ', '.join(f"{col} agtype" for col in columns)
            formatted_query = f"SELECT * FROM cypher('{GRAPH_NAME}', $${clean_query}$$) as ({column_def});"
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
        except Exception as e2:
            logger.error(f"Fallback query failed: {str(e2)}")
            st.error(f"Fallback query failed: {str(e2)}")
            return []
        finally:
            if 'cur' in locals(): cur.close()
            if 'conn' in locals(): conn.close()
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

- Relationships:
  - (TankContainer)-[:HAS_PREVIOUS_PRODUCT]->(Product) // Includes a sequence property (e.g., sequence: 1, 2, 3)
  - (TankContainer)-[:PART_OF]->(Fleetserie)
  - (TankContainer)-[:BELONGS_TO]->(Customer) // Note: May not be populated

- Notes:
  - TankContainer has prev1, prev2, prev3 fields storing previous product names as strings.
  - Use prev1, prev2, prev3 for queries about previous products unless HAS_PREVIOUS_PRODUCT is explicitly needed.
  - Product names should match against both 'name' and 'common_name' fields.
  - BELONGS_TO relationships may not exist in the graph.
  - Not all previous products (e.g., Specflex) are in the Product table, so rely on prev1, prev2, prev3 for such cases.
  - For fleet series queries, use PART_OF relationships but also consider TankContainer.fleetserie for robustness.
  - For previous product queries, return prev1, prev2, prev3 directly without AS aliases.
  - For all queries, avoid using AS aliases in the RETURN clause to ensure compatibility with Apache AGE.

RULES:
1. Use only the nodes and relationships defined in the schema.
2. Return a single, valid Cypher query as plain text (no markdown, no ```cypher``` blocks, no semicolons).
3. Always include a RETURN clause with specific fields (e.g., t.prev1, t.prev2, t.prev3 for previous products).
4. Use single quotes for string literals (e.g., 'HOYU 000001-3').
5. Ensure the query is syntactically correct for Apache AGE.
6. For previous product queries, use prev1, prev2, prev3 fields without AS aliases.
7. Handle cases where relationships may be missing by focusing on available data.

Question: "{question}"
""",
    input_variables=["question"],
)

def process_user_query(user_question):
    MAX_ATTEMPTS = 3
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        cypher_query = call_azure_openai([{
            "role": "user", 
            "content": cypher_prompt.format(question=user_question)
        }])
        if not cypher_query:
            attempts += 1
            continue
        if not validate_cypher_query(cypher_query):
            attempts += 1
            logger.warning(f"Invalid query, attempt {attempts}/{MAX_ATTEMPTS}: {cypher_query}")
            st.warning(f"Invalid query, attempt {attempts}/{MAX_ATTEMPTS}: {cypher_query}")
            continue
        result = run_cypher_query(cypher_query)
        if result:
            response = call_azure_openai([{
                "role": "user",
                "content": f"Explain these tank cleaning results in simple terms: {json.dumps(result)}. The user asked: '{user_question}'. Keep the response concise and focused on the key findings. For previous product queries, refer to the fields as Previous Product 1, Previous Product 2, and Previous Product 3 in the explanation."
            }])
        else:
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
            elif "fleet series" in user_question.lower():
                fleetserie_number = re.search(r'fleet series\s+(\d+)', user_question, re.IGNORECASE)
                if fleetserie_number:
                    fallback_cypher = f"MATCH (t:TankContainer) WHERE t.fleetserie = '{fleetserie_number.group(1)}' RETURN t.id, t.number, t.operator, t.capacity_l"
            if fallback_cypher:
                result = run_cypher_query(fallback_cypher)
                response = call_azure_openai([{
                    "role": "user",
                    "content": f"Explain these tank cleaning results in simple terms: {json.dumps(result)}. The user asked: '{user_question}'. Keep the response concise and focused on the key findings. For previous product queries, refer to the fields as Previous Product 1, Previous Product 2, and Previous Product 3 in the explanation."
                }]) if result else f"No results found for '{user_question}'. The graph may lack relevant data. Try questions about previous products or tank properties."
            else:
                response = f"No results found for '{user_question}'. The graph may lack relevant data. Try questions about previous products, fleet series, or cleaning codes."
        return cypher_query, result, response
    return None, None, f"Couldn't generate a valid query after {MAX_ATTEMPTS} attempts. Please try a different question."

# Streamlit UI
st.title("🛢️ Tank Cleaning Graph Analyzer")
st.markdown("""
This application queries a tank cleaning graph database using natural language.
It converts questions into Cypher queries using Azure OpenAI and executes them with Apache AGE.
""")

# Sidebar for initialization
with st.sidebar:
    st.header("Sample Questions")
    st.markdown("""
    Try asking:
    - Give me list of customer
    - What are the previous products in tank HOYU 000001-3?
    - Which tanks have capacity over 20000 liters?
    - Which tanks previously contained Methanol?
    - What cleaning agents are available for tank cleaning?
    """)
    st.header("📥 Import Data from CSV")
    table_select = st.selectbox("Select Table to Import", 
                              ["Customers", "EFTCO_Codes", "Fleetserie", "Products", "TankContainers"])
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file and st.button("Import CSV"):
        with st.spinner(f"Importing {table_select} data..."):
            import_csv_to_graph(table_select, uploaded_file)
    st.header("📤 Export Sample CSV")
    for table in ["Customers", "EFTCO_Codes", "Fleetserie", "Products", "TankContainers"]:
        st.download_button(
            label=f"Download {table} Sample",
            data=generate_sample_csv(table),
            file_name=f"{table}_sample.csv",
            mime="text/csv"
        )

# Verify data
with st.expander("Show Graph Status"):
    if st.button("Check Graph Contents"):
        with st.spinner("Checking graph..."):
            test_result = show_graph_status("MATCH (n) RETURN n LIMIT 1000")
            st.info(f"Nodes in graph: {len(test_result)}")
            if test_result:
                st.json(test_result)

# Query interface
st.header("Ask a Question")
user_question = st.text_input("Enter your question about tank cleaning:", 
                             placeholder="e.g., What are the previous products in tank HOYU 000001-3?")
if user_question:
    with st.spinner("Processing your question..."):
        cypher, result, answer = process_user_query(user_question)
        if cypher:
            # st.subheader("Generated Cypher Query")
            # st.code(cypher, language="cypher")
            # st.subheader("Query Results")
            # if result:
            #     st.json(result)
            # else:
            #     st.warning("No results returned from the query")
            st.subheader("Response")
            st.write(answer)
        else:
            st.error("Failed to generate a valid query. Please try a different question.")

# Footer
st.markdown("---")
st.caption("Tank Cleaning Graph Analyzer | Powered by Apache AGE and Azure OpenAI")