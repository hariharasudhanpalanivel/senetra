from clients.db import get_connection


def get_phcs():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            p.id,
            p.name,
            p.latitude,
            p.longitude,
            p.total_beds,
            p.total_doctors,
            p.total_nurses,
            d.name AS district_name
        FROM senetra.phcs p
        JOIN senetra.districts d
            ON p.district_id = d.id
        ORDER BY p.name
    """)

    data = cur.fetchall()

    cur.close()
    conn.close()

    return data


def get_phcs_by_district(district_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM senetra.phcs
        WHERE district_id = %s
        ORDER BY name
    """,
        (district_id,),
    )

    data = cur.fetchall()

    cur.close()
    conn.close()

    return data


def get_phc_by_id(phc_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM senetra.phcs
        WHERE id = %s
    """,
        (phc_id,),
    )

    data = cur.fetchone()

    cur.close()
    conn.close()

    return data
