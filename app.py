import os
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI
import pandas as pd
import io
import json

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
    "host": "localhost",
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
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_MODEL,
        messages=messages,
        max_tokens=500
    )    
    return response.choices[0].message.content.strip()

# Export sample CSVs
def generate_sample_csv(table_name):
    if table_name == "Customers":
        df = pd.DataFrame([{
            "Customer": "ChemGlobal Inc.",
            "ID": 1,
            "Address": "123 Industrial Rd",
            "Postalcode": "90210",
            "City": "Houston",
            "Country": "USA"
        }])
    elif table_name == "EFTCO_Codes":
        df = pd.DataFrame([
            {
                "Code": "C",
                "ID": 1,
                "Cleaning agent": "Cleaning agents",
                "Guideline": ""
            },
            {
                "Code": "C01",
                "ID": 2,
                "Cleaning agent": "Alkaline detergent",
                "Guideline": "Detergent with pH >7 used during the cleaning procedure..."
            }
        ])
    elif table_name == "Fleetserie":
        df = pd.DataFrame([{
            "Fleetserie number": 54001,
            "FleetserieID": 1,
            "Manufacturer": "Singamas",
            "Inner tank material": "Composite",
            "Insulation": False,
            "Insulation type": ""
        }])
    elif table_name == "Products":
        df = pd.DataFrame([
            {
                "Product": "Water",
                "ID": 1,
                "Manufacturer": "Universal Solvents Inc.",
                "Common name": "Water",
                "PH value": 7.0,
                "Water solutability": 0.0035,
                "Viscosity (mPas.s)": 0.89
            },
            {
                "Product": "Ethyl alcohol, Grain alcohol",
                "ID": 2,
                "Manufacturer": "ChemCo Ltd.",
                "Common name": "Ethanol",
                "PH value": 7.2,
                "Water solutability": 0.872,
                "Viscosity (mPas.s)": 1.2
            }
        ])
    elif table_name == "TankContainers":
        df = pd.DataFrame([
            {
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
            }
        ])
    else:
        df = pd.DataFrame()

    return df.to_csv(index=False)

def delete_existing_data(table_name):
    """Delete all nodes of the specified type from the graph"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        if table_name == "Customers":
            cypher = "MATCH (n:Customer) DETACH DELETE n"
        elif table_name == "EFTCO_Codes":
            cypher = "MATCH (n:EFTCOCode) DETACH DELETE n"
        elif table_name == "Fleetserie":
            cypher = "MATCH (n:Fleetserie) DETACH DELETE n"
        elif table_name == "Products":
            cypher = "MATCH (n:Product) DETACH DELETE n"
        elif table_name == "TankContainers":
            cypher = "MATCH (n:TankContainer) DETACH DELETE n"
        else:
            return False

        cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$) as (result agtype);")
        return True
    except Exception as e:
        st.error(f"Error deleting existing data for {table_name}: {e}")
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def import_csv_to_graph(table_name, uploaded_file):
    try:
        # First delete existing data for this table type
        if not delete_existing_data(table_name):
            st.error(f"Failed to delete existing {table_name} data")
            return

        df = pd.read_csv(uploaded_file)
        conn = psycopg2.connect(**POSTGRES_CONN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

        cypher_statements = []

        if table_name == "Customers":
            for _, row in df.iterrows():
                cypher_statements.append(
                    f"CREATE (:Customer {{id: '{row['ID']}', name: '{row['Customer'].replace("'", "''")}', address: '{row['Address'].replace("'", "''")}', postalcode: '{row['Postalcode']}', city: '{row['City'].replace("'", "''")}', country: '{row['Country'].replace("'", "''")}'}})"
                )
        elif table_name == "EFTCO_Codes":
            for _, row in df.iterrows():
                cypher_statements.append(
                    f"CREATE (:EFTCOCode {{code: '{row['Code']}', id: {row['ID']}, agent: '{row['Cleaning agent'].replace("'", "''")}', guideline: '{row['Guideline'].replace("'", "''")}'}})"
                )
        elif table_name == "Fleetserie":
            for _, row in df.iterrows():
                cypher_statements.append(
                    f"CREATE (:Fleetserie {{id: {row['FleetserieID']}, number: '{row['Fleetserie number']}', manufacturer: '{row['Manufacturer'].replace("'", "''")}', material: '{row['Inner tank material'].replace("'", "''")}', insulation: {str(row['Insulation']).lower()}, insulation_type: '{row['Insulation type'].replace("'", "''")}'}})"
                )
        elif table_name == "Products":
            for _, row in df.iterrows():
                cypher_statements.append(
                    f"CREATE (:Product {{id: {row['ID']}, name: '{row['Product'].replace("'", "''")}', manufacturer: '{row['Manufacturer'].replace("'", "''")}', common_name: '{row['Common name'].replace("'", "''")}', pH: {row['PH value']}, solubility: {row['Water solutability']}, viscosity: {row['Viscosity (mPas.s)']}}})"
                )
        elif table_name == "TankContainers":
            for _, row in df.iterrows():
                cypher_statements.append(
                    f"""CREATE (:TankContainer {{
                        id: '{row['TankID']}', 
                        number: '{row['Tank number']}',
                        fleetserie: '{row['Fleetserie number']}',
                        operator: '{row['Operator'].replace("'", "''")}',
                        type: '{row['Type']}',
                        manlid_holes: {row['Manlid holes']},
                        capacity_l: {row['Capacity (L)']},
                        baffles: {str(row['Baffles']).lower()},
                        coating: '{row['Coating'].replace("'", "''")}',
                        prev1: '{row['Previous Product 1'].replace("'", "''")}',
                        prev2: '{row['Previous Product 2'].replace("'", "''")}',
                        prev3: '{row['Previous Product 3'].replace("'", "''")}'
                    }})"""
                )
        
        for stmt in cypher_statements:
            try:
                cur.execute(f"SELECT * FROM cypher('{GRAPH_NAME}', $${stmt}$$) as (result agtype);")
            except Exception as e:
                st.error(f"Error executing statement: {stmt}\nError: {e}")
                continue

        st.success(f"Data imported successfully into {table_name}")
    except Exception as e:
        st.error(f"Error importing data for {table_name}: {e}")
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

def validate_cypher_query(cypher_query: str) -> bool:
    """Validate that the query uses only our allowed schema"""
    # First clean the query
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    
    allowed_terms = ["TankContainer", "CleanOrder", "Customer", "Product", 
                    "Fleetserie", "EFTCOCode", "EXECUTED_FOR", "BELONGS_TO",
                    "HAS_PRODUCT", "HAS_PREVIOUS_PRODUCT", "PART_OF", 
                    "USES_CLEANING_CODE"]
    forbidden_terms = ["Movie", "Person", "DIRECTED", "ACTED_IN"]
    
    return (any(term in cypher_query for term in allowed_terms) and 
            not any(term in cypher_query for term in forbidden_terms))

def run_cypher_query(cypher_query: str):
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
        
cypher_prompt = PromptTemplate(
    template="""Generate a Cypher query for Apache AGE graph database.

SCHEMA:
- Nodes:
  - TankContainer(id, number, type, capacity_l, operator)
  - CleanOrder(id, status)
  - Product(name, ph, solubility, viscosity)
  - Customer(id, name)
  - Fleetserie(number, manufacturer)
  - EFTCOCode(code, agent)

- Relationships:
  - (TankContainer)-[:BELONGS_TO]->(Customer)
  - (TankContainer)-[:HAS_PRODUCT]->(Product)
  - (TankContainer)-[:HAS_PREVIOUS_PRODUCT]->(Product)
  - (TankContainer)-[:PART_OF]->(Fleetserie)
  - (CleanOrder)-[:EXECUTED_FOR]->(TankContainer)
  - (CleanOrder)-[:USES_CLEANING_CODE]->(EFTCOCode)

RULES:
1. Only use nodes and relationships from the schema.
2. Return only valid Cypher (no markdown, no explanation).
3. Always include RETURN clause.
4. Use proper node labels and relationship types.

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
        
        # Clean and validate the query
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        st.write(f"Generated Cypher (before validation): `{cypher_query}`")
        
        if not validate_cypher_query(cypher_query):
            attempts += 1
            st.warning(f"Invalid query, attempt {attempts}/{MAX_ATTEMPTS}")
            continue
            
        # Execute the query
        result = run_cypher_query(cypher_query)
        
        # Generate explanation if we got results
        if result:
            response = call_azure_openai([{
                "role": "user",
                "content": f"Explain these tank cleaning results in simple terms: {result}. The user asked: '{user_question}'. Keep the response concise and focused on the key findings."
            }])
        else:
            response = "No results found for this query."
        
        return cypher_query, result, response
    
    return None, None, "Couldn't generate valid query after multiple attempts. Please try a different question."

# Streamlit UI
st.title("🛢️ Tank Cleaning Graph Analyzer")
st.markdown("""
This application helps you query a tank cleaning graph database using natural language.
It uses Azure OpenAI to convert your questions into Cypher queries and Apache AGE to execute them.
""")

# Sidebar for initialization
with st.sidebar:
    st.header("Sample Questions")
    st.markdown("""
    Try asking:
    - Which tanks contain ethanol?
    - List cleaning orders that used alkaline detergent
    - Which customer owns tank HOYU 000001-3?
    - Show tanks with capacity over 27000 liters
    - What are the previous products in tank HOYU 000001-3?
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
            test_result = run_cypher_query("MATCH (n) RETURN n LIMIT 10")
            st.info(f"Nodes in graph (sample of 10): {len(test_result)}")
            if test_result:
                st.json(test_result)

# Query interface
st.header("Ask a Question")
user_question = st.text_input("Enter your question about tank cleaning:", 
                             placeholder="e.g., Which tanks contain ethanol?")

if user_question:
    with st.spinner("Processing your question..."):
        cypher, result, answer = process_user_query(user_question)
        
        if cypher:
            st.subheader("Generated Cypher Query")
            st.code(cypher, language="cypher")
            
            st.subheader("Query Results")
            if result:
                st.json(result)
            else:
                st.warning("No results returned from the query")
            
            st.subheader("Explanation")
            st.write(answer)
        else:
            st.error("Failed to generate a valid query. Please try a different question.")

# Footer
st.markdown("---")
st.caption("Tank Cleaning Graph Analyzer | Powered by Apache AGE and Azure OpenAI")