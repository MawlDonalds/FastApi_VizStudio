# # database.py

# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import OperationalError
# from langchain_community.utilities import SQLDatabase
# from config import CONNECTION_STRING, DB_NAME, DB_HOST

# import pandas as pd

# _db_instance = None
# _engine_instance = None
# _tables = None

# def get_db_instance():
#     global _db_instance, _tables
#     if _db_instance is None:
#         try:
#             _db_instance = SQLDatabase.from_uri(CONNECTION_STRING)
#             _tables = _db_instance.get_usable_table_names()
#             print(f"✅ Berhasil terhubung ke database '{DB_NAME}' di {DB_HOST}")
#             print(f"📊 Tabel yang terdeteksi: {', '.join(_tables)}")
#         except OperationalError as e: 
#             print(f"❌ Gagal terhubung ke database: {str(e)}")
#             raise
#     return _db_instance

# def get_engine_instance():
#     global  _engine_instance
#     if _engine_instance is None:
#         try:
#             _engine_instance = create_engine()(CONNECTION_STRING)
            
#             with _engine_instance.connect() as conn:
#                 conn.execute(text("SELECT 1"))
#             print("✅ Berhasil menginisialisasi SQLAlchemy engine")
#         except OperationalError as e:
#             print(f"❌ Gagal menginisialisasi SQLAlchemy engine: {str(e)}")
#             raise
#     return _engine_instance

# def get_tables():
#     if _tables is None:
#         get_db_instance()
#     return _tables

# def  execute_sql_query(sql_query: str):
#     engine = get_engine_instance()
#     try:
#         with engine.connect() as conn:
#             df = pd.read_sql_query(text(sql_query), conn)
#         return df
#     except Exception as e:
#         raise Exception(f"Error executing SQL query: {str(e)}")