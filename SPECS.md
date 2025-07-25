# Init
- no additional work

# Configure
-> search dir and glob pattern
-> optional list of tables to use

# Create context
✅ create pl.SQLContext on connection (-> clear memory on exit)

# Tools (expected to perform computation and have side effects)
✅ execute SQL query (-> df as json)

# Resources (provide data but shouldn't perform significant computation or have side effects)
✅ get list of NWB files
✅ get available tables
✅ get table schema