from clients.db import get_connection


def get_countries():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, name, code
        FROM senetra.countries
        ORDER BY name
    """)

    data = cur.fetchall()

    cur.close()
    conn.close()

    return data


def get_country_by_id(country_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, name, code
        FROM senetra.countries
        WHERE id = %s
    """,
        (country_id,),
    )

    data = cur.fetchone()

    cur.close()
    conn.close()

    return data
