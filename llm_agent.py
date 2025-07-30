# llm_agent.py

import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.callbacks import BaseCallbackHandler
from typing import List, Dict, Any, Tuple, Optional

from config import GOOGLE_API_KEY
from database import get_db_instance

# Kelas untuk menangkap query SQL dan log
class SQLCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.sql_query = None
        self.logs = []
        self.all_queries = []
        self.raw_data = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.logs.append(f"=== TOOL START ===")
        self.logs.append(f"Serialized: {serialized}")
        self.logs.append(f"Input type: {type(input_str)}")
        self.logs.append(f"Input content: {input_str}")
        self.raw_data.append(('tool_start', serialized, input_str))
        
        query_found = self._extract_query_from_data(input_str)
        if query_found:
            self.logs.append(f"✓ Query found in tool_start: {query_found}")

    def on_tool_end(self, output, **kwargs):
        self.logs.append(f"=== TOOL END ===")
        self.logs.append(f"Output type: {type(output)}")
        self.logs.append(f"Output content: {output}")
        self.raw_data.append(('tool_end', output))
        
        query_found = self._extract_query_from_data(output)
        if query_found:
            self.logs.append(f"✓ Query found in tool_end: {query_found}")

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"=== AGENT ACTION ===")
        self.logs.append(f"Action: {action}")
        self.logs.append(f"Action type: {type(action)}")
        
        if hasattr(action, 'tool'):
            self.logs.append(f"Tool: {action.tool}")
        if hasattr(action, 'tool_input'):
            self.logs.append(f"Tool input type: {type(action.tool_input)}")
            self.logs.append(f"Tool input: {action.tool_input}")
            
            query_found = self._extract_query_from_data(action.tool_input)
            if query_found:
                self.logs.append(f"✓ Query found in agent_action: {query_found}")
        
        self.raw_data.append(('agent_action', action))

    def on_agent_end(self, output, **kwargs):
        self.logs.append(f"=== AGENT END ===")
        self.logs.append(f"Output: {output}")
        self.raw_data.append(('agent_end', output))

    def on_text(self, text, **kwargs):
        self.logs.append(f"=== TEXT CALLBACK ===")
        self.logs.append(f"Text: {text}")
        self.raw_data.append(('text', text))
        
        query_found = self._extract_query_from_data(text)
        if query_found:
            self.logs.append(f"✓ Query found in text: {query_found}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.logs.append(f"=== CHAIN START ===")
        self.logs.append(f"Serialized: {serialized}")
        self.logs.append(f"Inputs: {inputs}")
        self.raw_data.append(('chain_start', serialized, inputs))

    def on_chain_end(self, outputs, **kwargs):
        self.logs.append(f"=== CHAIN END ===")
        self.logs.append(f"Outputs: {outputs}")
        self.raw_data.append(('chain_end', outputs))

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logs.append(f"=== LLM START ===")
        self.logs.append(f"Prompts: {prompts}")
        self.raw_data.append(('llm_start', serialized, prompts))

    def on_llm_end(self, response, **kwargs):
        self.logs.append(f"=== LLM END ===")
        self.logs.append(f"Response: {response}")
        self.raw_data.append(('llm_end', response))
        
        if hasattr(response, 'generations'):
            for gen in response.generations:
                for g in gen:
                    if hasattr(g, 'text'):
                        query_found = self._extract_query_from_data(g.text)
                        if query_found:
                            self.logs.append(f"✓ Query found in llm_end: {query_found}")

    def _extract_query_from_data(self, data):
        queries_found = []
        
        if isinstance(data, dict):
            for key in ['query', 'sql', 'statement', 'command']:
                if key in data:
                    query = data[key]
                    if self._is_select_query(query):
                        queries_found.append(query)
                        self.sql_query = query
                        self.all_queries.append(query)
        
        elif isinstance(data, str):
            if self._is_select_query(data):
                queries_found.append(data)
                self.sql_query = data
                self.all_queries.append(data)
            
            patterns = [
                r'```sql\s*(.*?)\s*```',
                r'```\s*(SELECT.*?)\s*```',
                r'Query:\s*(SELECT.*?)(?=\n|$)',
                r'(SELECT\s+.*?FROM\s+.*?)(?=\n\n|\n(?=[A-Z])|$)',
                r'(SELECT\s+.*?FROM\s+\w+(?:\s+WHERE\s+.*?)?(?:\s+GROUP\s+BY\s+.*?)?(?:\s+ORDER\s+BY\s+.*?)?(?:\s+LIMIT\s+\d+)?)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, data, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    match = match.strip()
                    if self._is_select_query(match):
                        queries_found.append(match)
                        self.sql_query = match
                        self.all_queries.append(match)
        
        return queries_found[0] if queries_found else None

    def _is_select_query(self, text):
        if not isinstance(text, str) or len(text.strip()) < 10:
            return False
        
        text_upper = text.upper()
        return ("SELECT" in text_upper and 
                "FROM" in text_upper and 
                len(text.strip()) > 15)

    def get_sql_query(self):
        return self.sql_query

    def get_all_queries(self):
        return list(set(self.all_queries))

    def get_logs(self):
        return "\n".join(self.logs)
    
    def get_raw_data(self):
        return self.raw_data

# Inisialisasi LLM
_llm_instance = None

def get_llm_instance():
    global _llm_instance
    if _llm_instance is None:
        try:
            _llm_instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
            print("✅ LLM (Gemini 1.5 Flash) berhasil diinisialisasi.")
        except Exception as e:
            print(f"❌ Gagal menginisialisasi LLM: {str(e)}")
            raise
    return _llm_instance

# Buat SQL agent
def get_sql_agent(callback_handler: SQLCaptureCallback):
    llm = get_llm_instance()
    db = get_db_instance()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=True, # Set to False in production for less verbose logs
        callbacks=[callback_handler]
    )
    return agent

# Fungsi untuk mendeteksi kolom numerik, kategorikal, dan tanggal
def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    for col in categorical_cols.copy():
        if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
            try:
                pd.to_datetime(df[col].head(10))
                date_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass
    
    return numeric_cols, categorical_cols, date_cols

# Fungsi untuk membuat visualisasi berdasarkan prompt dan data
def create_smart_visualization(df: pd.DataFrame, prompt: str) -> List[Dict[str, Any]]:
    charts_data = []
    
    if df.empty:
        return charts_data
    
    # Ensure datetime type for 'sale_date' if present
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    
    numeric_cols, categorical_cols, date_cols = detect_column_types(df)
    prompt_lower = prompt.lower()
    
    # 1. Line chart untuk tren waktu (jika ada tanggal dan numerik)
    if date_cols and numeric_cols:
        try:
            for date_col in date_cols:
                df_time = df.groupby(date_col).agg({col: 'sum' for col in numeric_cols}).reset_index()
                for num_col in numeric_cols:
                    fig = px.line(df_time, x=date_col, y=num_col, 
                                    title=f"📈 Trend {num_col.replace('_', ' ').title()} Over Time",
                                    labels={date_col: 'Tanggal', num_col: num_col.replace('_', ' ').title()})
                    charts_data.append({
                        "chart_type": "line",
                        "data": fig.to_dict()['data'],
                        "layout": fig.to_dict()['layout'],
                        "title": fig.layout.title.text
                    })
        except Exception as e:
            print(f"⚠️ Error creating line chart: {str(e)}")
            pass
    
    # 2. Bar chart untuk data kategorikal dan numerik
    if categorical_cols and numeric_cols:
        try:
            for cat_col in categorical_cols[:2]:
                for num_col in numeric_cols[:2]:
                    df_cat = df.groupby(cat_col).agg({num_col: 'sum'}).reset_index()
                    df_cat = df_cat.sort_values(num_col, ascending=False).head(10)
                    fig = px.bar(df_cat, x=cat_col, y=num_col,
                                   title=f"📊 {num_col.replace('_', ' ').title()} per {cat_col.replace('_', ' ').title()}",
                                   labels={cat_col: cat_col.replace('_', ' ').title(), num_col: num_col.replace('_', ' ').title()})
                    charts_data.append({
                        "chart_type": "bar",
                        "data": fig.to_dict()['data'],
                        "layout": fig.to_dict()['layout'],
                        "title": fig.layout.title.text
                    })
        except Exception as e:
            print(f"⚠️ Error creating bar chart: {str(e)}")
            pass
    
    # 3. Pie chart untuk distribusi kategorikal (jika ada numerik dan data tidak terlalu banyak)
    if categorical_cols and numeric_cols and len(df) <= 20:
        try:
            for cat_col in categorical_cols[:1]:
                for num_col in numeric_cols[:1]:
                    df_cat = df.groupby(cat_col).agg({num_col: 'sum'}).reset_index()
                    df_cat = df_cat.sort_values(num_col, ascending=False).head(10)
                    fig = px.pie(df_cat, names=cat_col, values=num_col,
                                   title=f"🥧 Distribusi {num_col.replace('_', ' ').title()} per {cat_col.replace('_', ' ').title()}")
                    charts_data.append({
                        "chart_type": "pie",
                        "data": fig.to_dict()['data'],
                        "layout": fig.to_dict()['layout'],
                        "title": fig.layout.title.text
                    })
        except Exception as e:
            print(f"⚠️ Error creating pie chart: {str(e)}")
            pass
    
    # 4. Histogram untuk data numerik saja
    if numeric_cols:
        try:
            for num_col in numeric_cols[:2]:
                fig = px.histogram(df, x=num_col, nbins=20,
                                     title=f"📊 Distribusi {num_col.replace('_', ' ').title()}",
                                     labels={num_col: num_col.replace('_', ' ').title()})
                charts_data.append({
                    "chart_type": "histogram",
                    "data": fig.to_dict()['data'],
                    "layout": fig.to_dict()['layout'],
                    "title": fig.layout.title.text
                })
        except Exception as e:
            print(f"⚠️ Error creating histogram: {str(e)}")
            pass
            
    return charts_data if charts_data else []


# Fungsi untuk brute force mencari query SQL
def brute_force_extract_sql(data: Any, depth: int = 0) -> Optional[str]:
    """Fungsi rekursif untuk mencari query SQL di struktur data apapun"""
    if depth > 10:  # Prevent infinite recursion
        return None
    
    if isinstance(data, str):
        # Cek apakah string ini adalah query SQL
        if "SELECT" in data.upper() and "FROM" in data.upper() and len(data.strip()) > 15:
            return data.strip()
        
        # Coba regex patterns
        patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(SELECT.*?)\s*```',
            r'Query:\s*(SELECT.*?)(?=\n|$)',
            r'(SELECT\s+.*?FROM\s+.*?)(?=\n\n|\n(?=[A-Z])|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, data, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.strip()
                if "SELECT" in match.upper() and "FROM" in match.upper():
                    return match
    
    elif isinstance(data, dict):
        for key, value in data.items():
            result = brute_force_extract_sql(value, depth + 1)
            if result:
                return result
    
    elif isinstance(data, (list, tuple)):
        for item in data:
            result = brute_force_extract_sql(item, depth + 1)
            if result:
                return result
    
    elif hasattr(data, '__dict__'):
        for attr_name in dir(data):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(data, attr_name)
                    result = brute_force_extract_sql(attr_value, depth + 1)
                    if result:
                        return result
                except:
                    pass
    
    return None

# Fungsi untuk mencoba menjalankan query manual
def try_manual_query_generation(user_prompt: str) -> Optional[str]:
    """Coba generate query manual berdasarkan prompt"""
    try:
        llm = get_llm_instance()
        db = get_db_instance()
        manual_agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type="tool-calling",
            verbose=False # Keep this False for manual agent
        )
        
        specific_prompt = f"""
        Generate ONLY a SQL query for this request: {user_prompt}
        
        Return only the SQL query without any explanation or formatting.
        The query should be a valid PostgreSQL SELECT statement.
        """
        
        response = manual_agent.invoke({"input": specific_prompt})
        
        if isinstance(response, dict) and "output" in response:
            output = response["output"]
            extracted = brute_force_extract_sql(output)
            if extracted:
                return extracted
        
        return brute_force_extract_sql(response)
        
    except Exception as e:
        print(f"❌ Error in manual query generation: {str(e)}")
        return None