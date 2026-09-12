import psycopg2
from psycopg2.extras import RealDictCursor

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="senetra",
        user="postgres",
        password="password",
        cursor_factory=RealDictCursor
    )