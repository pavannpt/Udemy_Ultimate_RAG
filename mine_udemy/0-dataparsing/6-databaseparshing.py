import sqlite3
import os
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_core.documents import Document

os.makedirs("mine_udemy/data/databases", exist_ok=True)

conn = sqlite3.connect("mine_udemy/data/databases/company.db")
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY, name TEXT, role TEXT, department TEXT, salary REAL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY, name TEXT, status TEXT, budget REAL, lead_id INTEGER)''')

# Insert sample data
employees = [
    (1, 'John Doe', 'Senior Developer', 'Engineering', 95000),
    (2, 'Jane Smith', 'Data Scientist', 'Analytics', 105000),
    (3, 'Mike Johnson', 'Product Manager', 'Product', 110000),
    (4, 'Sarah Williams', 'DevOps Engineer', 'Engineering', 98000)
]

projects = [
    (1, 'RAG Implementation', 'Active', 150000, 1),
    (2, 'Data Pipeline', 'Completed', 80000, 2),
    (3, 'Customer Portal', 'Planning', 200000, 3),
    (4, 'ML Platform', 'Active', 250000, 2)
]

cursor.executemany("INSERT OR REPLACE INTO employees values (?,?,?,?,?)", employees)
cursor.executemany('INSERT OR REPLACE INTO projects VALUES (?,?,?,?,?)', projects)

#cursor.execute("Select * from employees")
#print(cursor.fetchall())


db = SQLDatabase.from_uri("sqlite:///mine_udemy/data/databases/company.db")
print(db.get_table_info())
print(db.get_table_info())

# Fetch data from the employees table
cursor = conn.cursor()
cursor.execute("SELECT * FROM employees")
employees_data = cursor.fetchall()

# Convert employees data into Langchain documents
employees_documents = [
    Document(page_content=str(employee), metadata={"id": employee[0], "name": employee[1], "role": employee[2], "department": employee[3], "salary": employee[4]})
    for employee in employees_data
]

# Fetch data from the projects table
cursor.execute("SELECT * FROM projects")
projects_data = cursor.fetchall()

# Convert projects data into Langchain documents
projects_documents = [
    Document(page_content=str(project), metadata={"id": project[0], "name": project[1], "status": project[2], "budget": project[3], "lead_id": project[4]})
    for project in projects_data
]

# Combine all documents
all_documents = employees_documents + projects_documents

print(all_documents)

cursor.close()
conn.commit()
conn.close()