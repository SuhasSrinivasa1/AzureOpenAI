# -*- coding: utf-8 -*-
"""
This script demonstrates how to connect to the Azure SQL database for GR DataLake.
It sets up the proxy environment and uses pyodbc to execute a simple query.

Usage:
    python db_connect.py

Notes:
    Replace the placeholder variables (SERVER, DATABASE, USERNAME, PASSWORD) with the actual credentials.
    - Proxy: It uses the HTTP_PROXY and HTTPS_PROXY environment variables to route traffic through a local proxy (127.0.0.1:3128).
    - ODBC Driver: The script uses the ODBC Driver 17 for SQL Server. If necessary, update this to the driver version installed on your machine.

"""

import os
import pyodbc

# Proxy configuration for Bosch environment
os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"

# Database connection parameters (replace with actual credentials)
SERVER = "grdatalake-mssql-qa-we-001.database.windows.net"
DATABASE = "grdatalake-sqldb-qa-we-001"
USERNAME = "<YOUR_DB_USERNAME>"
PASSWORD = "<YOUR_DB_PASSWORD>"

def connect_to_database():
    """Connect to the Azure SQL database and return the connection and cursor."""
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as e:
        print("Failed to connect to the database:", e)
        return None, None

def test_connection():
    """Execute a simple query to verify the connection."""
    conn, cursor = connect_to_database()
    if conn and cursor:
        try:
            cursor.execute("SELECT GETDATE()")
            result = cursor.fetchone()
            print("Current database datetime:", result[0])
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    test_connection()
