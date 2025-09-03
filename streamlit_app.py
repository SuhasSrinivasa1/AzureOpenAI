# streamlit_app.py  — Streamlit 1.12 compatible
# Azure OpenAI Chat + Azure SQL (NL→SQL), with guided options and durable results

import os
import re
import json
import datetime
import traceback
from typing import Optional, List, Dict

import streamlit as st
import requests
import pyodbc
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Proxy (Bosch)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:3128"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"

# ──────────────────────────────────────────────────────────────────────────────
# Azure OpenAI
# ──────────────────────────────────────────────────────────────────────────────
ENDPOINT    = "https://ijj3kor-7111-resource.cognitiveservices.azure.com/"
API_VERSION = "2024-12-01-preview"
DEPLOYMENT  = "gpt-4.1"
API_KEY     = "4PV2I3zS721qEgLi3JQ5VA62D3pCa5LHumYf3gJMOjzOer8BkULYJQQJ99BHACHYHv6XJ3w3AAAAACOGEvmc"

OA_URL     = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
OA_HEADERS = {"Content-Type": "application/json", "api-key": API_KEY}

def ask_azure_openai(messages: List[Dict[str, str]]) -> str:
    payload = {"messages": messages, "max_tokens": 4096, "temperature": 0.7}
    r = requests.post(OA_URL, headers=OA_HEADERS, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]

# ──────────────────────────────────────────────────────────────────────────────
# Azure SQL (ODBC 17, AAD password)
# ──────────────────────────────────────────────────────────────────────────────
ODBC_DRIVER = "ODBC Driver 17 for SQL Server"
SERVER      = "grdatalake-mssql-qa-we-001.database.windows.net"
DATABASE    = "grdatalake-sqldb-qa-we-001"
USERNAME    = "ijj3kor@bosch.com"         # AAD UPN
PASSWORD    = "animalPlanet919"           # AAD password (no MFA)

def connect_to_database():
    conn_str = (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={SERVER};DATABASE={DATABASE};"
        f"UID={USERNAME};PWD={PASSWORD};"
        "Authentication=ActiveDirectoryPassword;"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)

def is_select_only(sql: str) -> bool:
    stripped = re.sub(r"^\s*(--[^\n]*\n|/\*.*?\*/\s*)*", "", sql, flags=re.S)
    return stripped.strip().lower().startswith("select")

def run_sql(sql: str, limit_rows: Optional[int] = None) -> pd.DataFrame:
    with connect_to_database() as conn:
        df = pd.read_sql(sql, conn)
    if limit_rows is not None and len(df) > limit_rows:
        return df.head(limit_rows)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Discovery & feedback helpers (Streamlit 1.12 caching)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def list_schemas() -> List[str]:
    sql = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA ORDER BY SCHEMA_NAME;"
    with connect_to_database() as conn:
        df = pd.read_sql(sql, conn)
    return df["SCHEMA_NAME"].tolist()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def list_tables_in_schema(schema_name: str) -> pd.DataFrame:
    sql = """
      SELECT TABLE_NAME, TABLE_TYPE
      FROM INFORMATION_SCHEMA.TABLES
      WHERE TABLE_SCHEMA = ?
      ORDER BY TABLE_NAME;
    """
    with connect_to_database() as conn:
        return pd.read_sql(sql, conn, params=[schema_name])

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def list_columns(schema_name: str, table_name: str) -> pd.DataFrame:
    sql = """
      SELECT COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
      FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
      ORDER BY ORDINAL_POSITION;
    """
    with connect_to_database() as conn:
        return pd.read_sql(sql, conn, params=[schema_name, table_name])

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_archibus_tables() -> List[str]:
    sql = """
      SELECT TABLE_NAME
      FROM INFORMATION_SCHEMA.TABLES
      WHERE TABLE_SCHEMA = 'ARCHIBUS'
      ORDER BY TABLE_NAME;
    """
    with connect_to_database() as conn:
        return pd.read_sql(sql, conn)["TABLE_NAME"].tolist()

def get_columns_for_table(table_name: str) -> pd.DataFrame:
    return list_columns("ARCHIBUS", table_name)

def sample_distinct_values(table_name: str, col: str, limit: int = 200) -> List[str]:
    sql = f"SELECT DISTINCT TOP {int(limit)} [{col}] AS v FROM ARCHIBUS.[{table_name}] WHERE [{col}] IS NOT NULL ORDER BY 1"
    with connect_to_database() as conn:
        df = pd.read_sql(sql, conn)
    return df["v"].astype(str).tolist()

def log_feedback_jsonl(payload: dict, path: str = "nl2sql_feedback.jsonl") -> None:
    payload = dict(payload)
    payload["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Azure OpenAI + Azure SQL", layout="wide")
st.title("Azure OpenAI Chat + Azure SQL Query")

tab_chat, tab_sql = st.tabs(["💬 Chat with Azure OpenAI (NL→SQL)", "🗄️ SQL Query (Azure SQL DB)"])

# ──────────────────────────────────────────────────────────────────────────────
# Chat tab: Guided NL→SQL + persistent result + feedback logging
# ──────────────────────────────────────────────────────────────────────────────
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []  # visible transcript
    if "last_generated_sql" not in st.session_state:
        st.session_state.last_generated_sql = ""
    if "last_exec_ok" not in st.session_state:
        st.session_state.last_exec_ok = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "last_df_dict" not in st.session_state:  # store df persistently
        st.session_state.last_df_dict = {"records": [], "columns": []}
    if "last_rowcount" not in st.session_state:
        st.session_state.last_rowcount = None

    st.subheader("Conversation")
    for msg in st.session_state.history:
        role = "User" if msg["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {msg['content']}")

    st.markdown("---")
    st.markdown("**Ask in natural language — I’ll generate a safe SELECT and query the QA DB.**")
    auto_run_sql = st.checkbox("Auto-run generated SQL", value=True)
    row_cap = st.number_input("Max rows to return", value=1000, min_value=10, step=10)

    # Model hints expander
    with st.expander("Model hints (recommended) — help me pick the right table/columns/filters"):
        try:
            tables = get_archibus_tables()
        except Exception as e:
            tables = []
            st.warning(f"Couldn’t list ARCHIBUS tables: {e}")

        primary_table = st.selectbox("Primary table (ARCHIBUS)", options=["(none)"] + tables, index=0)
        prop_col = city_col = addl_filter_col = addl_filter_val = ""
        synonyms_flag = st.checkbox("Treat 'Bangalore' and 'Bengaluru' as the same city", value=True)

        cols = []
        if primary_table and primary_table != "(none)":
            try:
                cols_df = get_columns_for_table(primary_table)
                cols = cols_df["COLUMN_NAME"].tolist()
            except Exception as e:
                st.warning(f"Couldn’t list columns: {e}")

            def pick(defaults: List[str], pool: List[str]) -> str:
                for d in defaults:
                    if d in pool: return d
                return pool[0] if pool else ""

            prop_col = st.selectbox(
                "Property ID column", options=["(none)"] + cols,
                index=(["(none)"] + cols).index(pick(["pr_id","property_id","prid"], cols)) if cols else 0
            )
            city_col = st.selectbox(
                "City column", options=["(none)"] + cols,
                index=(["(none)"] + cols).index(pick(["city","city_id","city_name"], cols)) if cols else 0
            )

            quick_vals = []
            if city_col and city_col != "(none)":
                try:
                    quick_vals = sample_distinct_values(primary_table, city_col, limit=50)
                except Exception:
                    quick_vals = []
            if quick_vals:
                st.caption("Distinct sample values for chosen city column:")
                st.write(", ".join(quick_vals[:25]))

            addl_filter_col = st.text_input("Additional filter column (optional)")
            addl_filter_val = st.text_input("Additional filter value (optional)")

    extra_context = st.text_input("Additional instructions for the model (optional)")

    user_prompt = st.text_input("Type your message, e.g.: 'Count properties (pr_id) in Bengaluru from ARCHIBUS'")
    send = st.button("Send")

    if send and user_prompt.strip():
        # Build schema hints
        schema_hints = []
        if primary_table and primary_table != "(none)":
            schema_hints.append(f"Use ARCHIBUS.[{primary_table}] as the primary table.")
            if prop_col and prop_col != "(none)":
                schema_hints.append(f"Property ID column is [{prop_col}].")
            if city_col and city_col != "(none)":
                schema_hints.append(f"City column is [{city_col}].")
        if synonyms_flag:
            schema_hints.append("Treat 'Bangalore' and 'Bengaluru' as synonyms.")
        if addl_filter_col and addl_filter_val:
            schema_hints.append(f"Apply filter: [{addl_filter_col}] = '{addl_filter_val}'.")
        if extra_context:
            schema_hints.append(f"Extra user context: {extra_context}")

        schema_text = "\n".join(schema_hints) if schema_hints else "No schema hints provided."

        system_instructions = (
            "You convert a user question into ONE safe T-SQL SELECT for Azure SQL.\n"
            "RULES:\n"
            "1) SELECT-only. Forbid INSERT/UPDATE/DELETE/MERGE/TRUNCATE/DROP/ALTER; no multi-statement; no GO.\n"
            "2) Query ONLY from the ARCHIBUS schema.\n"
            "3) Use explicit columns; add TOP/FETCH when returning detail rows.\n"
            "4) If counting unique IDs, prefer COUNT(DISTINCT ...).\n"
            "5) Follow schema hints exactly when provided.\n"
            "6) Output ONLY SQL in a fenced block:\n```sql\nSELECT ...\n```"
        )

        try:
            history_for_sql = [
                {"role": "system", "content": system_instructions},
                {"role": "system", "content": f"Schema hints:\n{schema_text}"},
                {"role": "user", "content": user_prompt},
            ]
            with st.spinner("Generating SQL from your question..."):
                sql_reply = ask_azure_openai(history_for_sql)
        except Exception as e:
            st.error(f"OpenAI (SQL generation) failed: {e}")
            st.code(traceback.format_exc())
            sql_reply = ""

        m = re.search(r"```sql\s*(.*?)```", sql_reply, flags=re.S | re.I)
        generated_sql = m.group(1).strip() if m else sql_reply.strip()

        # Safety checks
        def _is_select_only(s):
            s2 = re.sub(r"^\s*(--[^\n]*\n|/\*.*?\*/\s*)*", "", s, flags=re.S)
            return s2.strip().lower().startswith("select")
        def _single_stmt(s):
            low = " " + s.lower() + " "
            return ("; " not in low) and ("\n;\n" not in low) and (" go " not in low)
        def _archibus_only(s):
            low = s.lower()
            return "archibus." in low and not any(b in low for b in [" msdb."," master."," tempdb."," sys."," information_schema."])

        safe = True
        safe_msgs = []
        if not generated_sql:
            safe = False; safe_msgs.append("No SQL produced.")
        if safe and not _is_select_only(generated_sql):
            safe = False; safe_msgs.append("Not a SELECT.")
        if safe and not _single_stmt(generated_sql):
            safe = False; safe_msgs.append("Multiple statements.")
        if safe and not _archibus_only(generated_sql):
            safe = False; safe_msgs.append("Outside ARCHIBUS schema.")

        if safe and ("count(" not in generated_sql.lower()) and (" top " not in generated_sql.lower()) and (" fetch " not in generated_sql.lower()):
            generated_sql = re.sub(r"(?i)^select\s+", f"SELECT TOP {int(row_cap)} ", generated_sql, count=1)

        # Persist generated SQL
        st.session_state.last_generated_sql = generated_sql

        # Execute (and persist result) if allowed
        exec_ok = False
        rowcount = None
        error_text = ""
        df = pd.DataFrame()
        if safe and auto_run_sql and generated_sql:
            try:
                with st.spinner("Running SQL against QA DB..."):
                    df = run_sql(generated_sql, limit_rows=int(row_cap))
                exec_ok = True
                rowcount = len(df)
                st.success(f"Returned {rowcount} row(s).")
            except Exception as e:
                error_text = f"{e}"
                st.error(f"Query failed: {e}")
                st.code(traceback.format_exc())

        # Persist execution outcome and dataframe so it stays visible
        st.session_state.last_exec_ok  = exec_ok
        st.session_state.last_rowcount = rowcount
        st.session_state.last_error    = error_text
        if exec_ok:
            st.session_state.last_df_dict = {"records": df.to_dict(orient="records"), "columns": list(df.columns)}

        # Append to chat transcript (visible)
        st.session_state.history.append({"role": "user", "content": user_prompt})
        st.session_state.history.append({"role": "assistant",
                                         "content": "SQL generated and executed (see below)." if exec_ok
                                                    else "SQL generated. Execution failed or skipped."})

    # Always show the most recent generated SQL and result (persistent)
    if st.session_state.last_generated_sql:
        st.markdown("**Generated SQL:**")
        st.code(st.session_state.last_generated_sql, language="sql")

    if st.session_state.last_exec_ok:
        # reconstruct DF from session state
        df_show = pd.DataFrame.from_records(st.session_state.last_df_dict["records"],
                                            columns=st.session_state.last_df_dict["columns"])
        st.dataframe(df_show, height=420)
    elif st.session_state.last_error:
        st.error(st.session_state.last_error)

    # Feedback box (writes JSONL for training/RAG)
    st.markdown("---")
    st.markdown("**Help improve results:**")
    feedback_text  = st.text_area("Any corrections or notes? (optional)", height=90, key="fb_text")
    corrected_sql  = st.text_area("If you know the correct SQL, paste it here (optional)", height=90, key="fb_sql")
    if st.button("💾 Save Feedback"):
        record = {
            "user_prompt": (st.session_state.history[-2]["content"] if len(st.session_state.history) >= 2 else ""),
            "schema_hints": [],  # you can also store the hints you passed above if needed
            "generated_sql": st.session_state.last_generated_sql,
            "executed": st.session_state.last_exec_ok,
            "rowcount": st.session_state.last_rowcount,
            "error": st.session_state.last_error,
            "feedback_text": feedback_text,
            "corrected_sql": corrected_sql,
        }
        try:
            log_feedback_jsonl(record)
            st.success("Feedback saved to nl2sql_feedback.jsonl")
        except Exception as e:
            st.error(f"Could not save feedback: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# SQL tab: Schema + Table/Views pickers, editor, and runner (no nested columns)
# ──────────────────────────────────────────────────────────────────────────────
with tab_sql:
    st.subheader("Run SQL against Azure SQL DB")

    try:
        schemas = list_schemas()
    except Exception as e:
        schemas = []
        st.error(f"Could not fetch schemas: {e}")

    default_schema_idx = schemas.index("ARCHIBUS") if "ARCHIBUS" in schemas else (0 if schemas else 0)
    schema_choice = st.selectbox("Schema", options=schemas or ["(none)"], index=default_schema_idx)

    tables_df = pd.DataFrame(columns=["TABLE_NAME", "TABLE_TYPE"])
    table_options = []
    if schema_choice and schema_choice != "(none)":
        try:
            tables_df = list_tables_in_schema(schema_choice)
            table_options = [
                f"{row.TABLE_NAME} ({'Table' if row.TABLE_TYPE=='BASE TABLE' else 'View'})"
                for _, row in tables_df.iterrows()
            ]
        except Exception as e:
            st.error(f"Could not fetch objects for schema '{schema_choice}': {e}")

    table_label = st.selectbox("Table/Views", options=table_options or ["(none)"], index=0 if table_options else 0)
    selected_table = table_label.split(" (")[0] if table_options and table_label != "(none)" else ""

    if selected_table:
        with st.expander(f"Columns in {schema_choice}.{selected_table}"):
            try:
                cols_df = list_columns(schema_choice, selected_table)
                st.dataframe(cols_df, height=220)
            except Exception as e:
                st.warning(f"Could not list columns: {e}")

    if "sql_editor" not in st.session_state:
        st.session_state.sql_editor = "SELECT TOP 50 name, create_date, modify_date FROM sys.tables ORDER BY modify_date DESC;"

    if selected_table and st.button("Insert SELECT template"):
        st.session_state.sql_editor = f"SELECT TOP 50 * FROM [{schema_choice}].[{selected_table}];"

    sql_editor_val = st.text_area("SQL", value=st.session_state.sql_editor, height=180,
                                  help="SELECT-only by default for safety.")
    if sql_editor_val != st.session_state.sql_editor:
        st.session_state.sql_editor = sql_editor_val

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        limit_rows2 = st.number_input("Row limit", value=1000, min_value=10, step=10, key="limit_rows2")
    with ctrl2:
        allow_non_select = st.checkbox("Allow non-SELECT (DML/DDL)", value=False,
                                       help="Leave off to block INSERT/UPDATE/DELETE/TRUNCATE/DROP, etc.")
    with ctrl3:
        run_btn = st.button("▶️ Run Query")

    with st.expander("Connection details"):
        st.text_input("Server", value=SERVER, disabled=True)
        st.text_input("Database", value=DATABASE, disabled=True)
        st.text_input("User (AAD UPN)", value=USERNAME, disabled=True)
        if st.button("Test Connection"):
            try:
                with connect_to_database() as conn:
                    cur = conn.cursor(); cur.execute("SELECT GETDATE()")
                    dt = cur.fetchone()[0]
                st.success(f"Connected! Database time: {dt}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
                st.code(traceback.format_exc())

    if run_btn:
        if not allow_non_select and not is_select_only(st.session_state.sql_editor):
            st.warning("Only SELECT statements are allowed. Tick 'Allow non-SELECT (DML/DDL)' to proceed.")
        else:
            with st.spinner("Running query..."):
                try:
                    if is_select_only(st.session_state.sql_editor) or not allow_non_select:
                        df2 = run_sql(st.session_state.sql_editor, limit_rows=limit_rows2)
                        st.success(f"Returned {len(df2)} row(s).")
                        st.dataframe(df2, height=420)
                    else:
                        with connect_to_database() as conn:
                            cur = conn.cursor()
                            cur.execute(st.session_state.sql_editor)
                            rowcount = cur.rowcount
                            conn.commit()
                        st.success(f"Statement executed. Row count: {rowcount}")
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    st.code(traceback.format_exc())

st.caption("Streamlit 1.12 build: chat_input not used; results persist across reruns.")
