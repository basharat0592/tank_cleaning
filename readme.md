## 📄 `README.md`

````markdown
# 🛢️ Tank Cleaning Graph Analyzer

**Tank Cleaning Graph Analyzer** is a Streamlit-based app that allows users to visualize, query, and analyze tank cleaning data stored in a graph database using [Apache AGE](https://github.com/apache/age) (PostgreSQL extension for Cypher support). The app integrates with Azure OpenAI to generate Cypher queries using natural language and supports CSV data import/export for several domain-specific entities.

---

## 🚀 Features

- 🔎 Natural language to Cypher query generation using Azure OpenAI
- 🗃️ Support for importing/exporting structured CSVs into Apache AGE
- 🧠 Graph schema-aware validation and execution
- 🧪 Example entity types: Customers, Products, EFTCO Codes, Fleetserie, Tank Containers
- 🐘 Seamless integration with PostgreSQL and Apache AGE
- 🌐 Streamlit UI for interaction and result visualization

---

## 🏗️ Architecture Overview

- **Frontend**: Streamlit
- **Backend**: Python, psycopg2 for PostgreSQL
- **Graph DB**: Apache AGE (Cypher on PostgreSQL)
- **LLM**: Azure OpenAI (Chat Completion API)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tank_cleaning.git
cd tank_cleaning
````

### 2. Create `.env` file

```env
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_API_VERSION=2023-05-15
AZURE_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_DEPLOYMENT_MODEL=gpt-35-turbo
```

### 3. Set up PostgreSQL with Apache AGE

```bash
# Start PostgreSQL with AGE (use Docker or local install)
# Ensure the 'age' extension is created and loaded
CREATE EXTENSION age;
LOAD 'age';
SELECT create_graph('tank_cleaning_graph');
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📁 CSV Import Support

Supported tables:

* Customers
* EFTCO\_Codes
* Fleetserie
* Products
* TankContainers

Each import deletes previous nodes of the same type and re-creates them in the graph.

---

## 💬 Prompt to Cypher

Azure OpenAI is used to convert natural language queries into Cypher queries. The app validates the schema to avoid misuse or hallucinated queries.

---

## 🧪 Sample Query

> “Show all tank containers with capacity over 22000 liters and manufactured by Singamas”

Produces a Cypher query like:

```cypher
MATCH (t:TankContainer)
WHERE t.capacity_l > 22000 AND t.operator = 'Singamas'
RETURN t
```
---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📬 Contact

For questions, reach out to \[[basharathussain05@gmail.com](mailto:basharathussain05@gmail.com)].
