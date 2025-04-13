################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', '<>')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

class Schema:
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)
        self._subquery_schemas = {}  # New: Map subquery aliases to their output columns

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    @property
    def subquery_schemas(self):
        return self._subquery_schemas

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1
        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1
        return idMap

    def add_subquery_schema(self, alias, columns):
        """Add columns available from a subquery under its alias."""
        self._subquery_schemas[alias] = [col.lower() for col in columns]
def get_schema(db):
    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    print(f"Schema: {schema}")  # Debug print
    return schema

def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols
    return schema

def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1:qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = []
    current_tok = ""
    i = 0
    while i < len(string):
        char = string[i]
        if char.isspace():
            if current_tok:
                toks.append(current_tok.lower())
                current_tok = ""
            i += 1
        elif char in "=><!(),;":
            if current_tok:
                # Check if current_tok ends with '.' and next char is '*'
                if current_tok.endswith('.') and char == '*':
                    toks.append(current_tok + '*')  # Combine 's.' and '*' into 's.*'
                    current_tok = ""
                    i += 1
                    continue
                toks.append(current_tok.lower())
                current_tok = ""
            if char == '<' and i + 1 < len(string) and string[i + 1] == '>':
                toks.append('<>')
                i += 2
            elif char == '>' and i + 1 < len(string) and string[i + 1] == '=':
                toks.append('>=')
                i += 2
            elif char == '<' and i + 1 < len(string) and string[i + 1] == '=':
                toks.append('<=')
                i += 2
            elif char == '!' and i + 1 < len(string) and string[i + 1] == '=':
                toks.append('!=')
                i += 2
            else:
                toks.append(char)
                i += 1
        else:
            current_tok += char
            i += 1
    if current_tok:
        toks.append(current_tok.lower())

    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    print(f"Tokenized: {toks}")
    return toks

def scan_alias(toks):
    alias = {}
    i = 0
    while i < len(toks) - 1:
        if toks[i].lower() == 'as' and i > 0 and i + 1 < len(toks):
            # Check if this is a table alias after 'FROM' or 'JOIN'
            j = i - 1
            while j >= 0 and toks[j] in ('(', ')'):
                j -= 1
            if j >= 0 and toks[j-1].lower() in ('from', 'join'):
                print(f"Explicit alias: {toks[i+1]} -> {toks[i-1]}")
                alias[toks[i + 1]] = toks[i - 1]
            # Handle subquery aliases by not assigning a table yet
            elif toks[j] == ')' and j > 0 and toks[j-1].lower() == 'select':
                print(f"Subquery alias: {toks[i+1]} (no table assigned)")
                alias[toks[i + 1]] = None  # Subquery alias, no table
            i += 2
            continue
        # Implicit alias after FROM or JOIN
        if i > 0 and toks[i-1].lower() in ['from', 'join']:
            if (i + 1 < len(toks) and 
                toks[i+1].lower() not in ['where', 'group', 'order', 'having', 'limit', 'union', 'intersect', 'except', 'and', 'or', 'join', 'on', 'as', 'using'] and
                toks[i] != '(' and 
                toks[i+1].isalnum()):
                print(f"Implicit alias: {toks[i+1]} -> {toks[i]}")
                alias[toks[i+1]] = toks[i]
                i += 2
                continue
        i += 1
    print(f"scan_alias result: {alias}")
    return alias

def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    # Add schema tables only if not already present as keys
    for key in schema:
        if key not in tables.keys():
            tables[key] = key
    return tables

def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    tok = toks[start_idx]
    print(f"Parsing column: '{tok}' at {start_idx}, default_tables: {default_tables}")

    if tok.lower() in WHERE_OPS or tok.lower() in CLAUSE_KEYWORDS or tok.lower() in JOIN_KEYWORDS or tok.lower() in COND_OPS:
        raise ValueError(f"Error col: {tok} - token is an operator, not a column")

    # Handle wildcard with alias (e.g., 's.*')
    if tok.endswith('.*'):
        alias = tok[:-2]  # Remove '.*' to get the alias
        table_or_subquery = tables_with_alias.get(alias, alias)
        # Check if alias is valid
        if table_or_subquery in schema.schema or table_or_subquery in schema.subquery_schemas:
            return start_idx + 1, f"__{table_or_subquery}.*__"  # Use a special identifier for wildcard
        else:
            print(f"Warning: Alias '{alias}' not found in tables or subqueries. Using placeholder.")
            return start_idx + 1, f"__{alias}.*__"

    if tok == "*":
        return start_idx + 1, schema.idMap[tok]
    if '.' in tok:
        alias, col = tok.split('.')
        table_or_subquery = tables_with_alias.get(alias, alias)
        col_lower = col.lower()
        
        if table_or_subquery in schema.subquery_schemas:
            if col_lower in schema.subquery_schemas[table_or_subquery]:
                return start_idx + 1, f"__{table_or_subquery}.{col_lower}__"
            print(f"Warning: Column '{col}' not found in subquery '{table_or_subquery}'. Proceeding with placeholder.")
            return start_idx + 1, f"__{table_or_subquery}.{col_lower}__"
        
        key = table_or_subquery + "." + col
        key_lower = key.lower()
        if key_lower in schema.idMap:
            return start_idx + 1, schema.idMap[key_lower]
        print(f"Warning: Column '{tok}' not found in schema for table '{table_or_subquery}'. Using placeholder identifier.")
        return start_idx + 1, f"__{table_or_subquery}.{col_lower}__"

    assert default_tables is not None and len(default_tables) > 0, "Default tables required"
    tok_lower = tok.lower()
    possible_keys = []

    for alias in default_tables:
        table = tables_with_alias.get(alias, alias)
        if table in schema.subquery_schemas:
            cols = schema.subquery_schemas[table]
            if tok_lower in cols:
                return start_idx + 1, f"__{table}.{tok_lower}__"
        else:
            cols = schema.schema.get(table, [])
            for col in cols:
                if tok_lower == col.lower():
                    key = table + "." + col
                    possible_keys.append(key)

    if not possible_keys and len(default_tables) < len(tables_with_alias):
        for alias, table in tables_with_alias.items():
            if alias in default_tables:
                continue
            if table in schema.subquery_schemas:
                cols = schema.subquery_schemas[table]
                if tok_lower in cols:
                    return start_idx + 1, f"__{table}.{tok_lower}__"
            else:
                cols = schema.schema.get(table, [])
                for col in cols:
                    if tok_lower == col.lower():
                        key = table + "." + col
                        possible_keys.append(key)

    if len(possible_keys) == 1:
        key_lower = possible_keys[0].lower()
        if key_lower in schema.idMap:
            return start_idx + 1, schema.idMap[key_lower]
        print(f"Warning: Key '{key_lower}' not in idMap. Using placeholder.")
        return start_idx + 1, f"__{possible_keys[0].lower()}__"
    elif len(possible_keys) > 1:
        chosen_key = possible_keys[0]
        print(f"Warning: Ambiguous column '{tok}' matches {possible_keys}. Defaulting to '{chosen_key}'")
        key_lower = chosen_key.lower()
        if key_lower in schema.idMap:
            return start_idx + 1, schema.idMap[key_lower]
        return start_idx + 1, f"__{key_lower}__"
    print(f"Warning: Column '{tok}' not resolved. Assigning placeholder with first default table '{default_tables[0]}'.")
    return start_idx + 1, f"__{tables_with_alias[default_tables[0]]}.{tok_lower}__"

def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1
    return idx, (agg_id, col_id, isDistinct)
def parse_expression(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """Parse an expression, including comparisons, inside aggregations."""
    idx = start_idx
    len_ = len(toks)

    # Parse the left-hand side of the expression
    idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)

    # Check for a comparison operator
    if idx < len_ and toks[idx] in WHERE_OPS:
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        return idx, {'expr': (op_id, val_unit, val1)}
    
    # If no comparison operator, return the val_unit as is
    return idx, val_unit
def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # Parse the first value or aggregation
    if toks[idx] in AGG_OPS and toks[idx] != 'none':
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '(', f"Expected '(' after {toks[idx-1]}"
        idx += 1
        isDistinct = False
        if toks[idx] == 'distinct':
            isDistinct = True
            idx += 1
        idx, val = parse_expression(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')', f"Expected ')' after aggregation, got {toks[idx]}"
        idx += 1
        col_unit1 = (agg_id, val, isDistinct)
        result = (UNIT_OPS.index('none'), col_unit1, None)
    else:
        # Handle nested parentheses or simple values
        if idx < len_ and toks[idx] == '(':
            idx, sub_result = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            result = sub_result
        else:
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            if isinstance(val1, tuple) and len(val1) == 3:
                col_unit1 = val1
            else:
                col_unit1 = (AGG_OPS.index("none"), val1, False)
            result = (UNIT_OPS.index('none'), col_unit1, None)

    # Handle arithmetic operations outside the initial value
    while idx < len_ and toks[idx] in UNIT_OPS + ('||',):
        if toks[idx] == '||':
            unit_op = UNIT_OPS.index('+')  # Treat '||' as concatenation
        else:
            unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        # Parse the right-hand side, which could be another nested expression
        if idx < len_ and toks[idx] == '(':
            idx, val2 = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        else:
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = (AGG_OPS.index("none"), val2, False) if not isinstance(val2, tuple) or len(val2) != 3 else val2
        col_unit2 = val2 if isinstance(val2, tuple) and len(val2) == 3 else (AGG_OPS.index("none"), val2, False)
        result = (unit_op, result, col_unit2)

    if isBlock:
        assert idx < len_ and toks[idx] == ')', f"Expected closing ')' for block at {idx}, got {toks[idx]}"
        idx += 1

    return idx, result

def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]
    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1
    return idx, schema.idMap[key], key

def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # Expanded list of SQLite functions
    SQL_FUNCTIONS = [
        'date', 'time', 'datetime', 'strftime', 'julianday',  # Date/time functions
        'current_timestamp', 'current_date', 'current_time',
        'abs', 'length', 'lower', 'upper', 'round', 'trim',  # Common scalar functions
        'coalesce', 'nullif', 'substr', 'instr', 'replace'   # Additional useful functions
    ]

    # Handle SQL functions (including nested ones)
    if (toks[idx].lower() in SQL_FUNCTIONS or 
        (idx + 1 < len_ and toks[idx + 1] == '(')):  # Generic function detection
        func_name = toks[idx].lower()
        idx += 1  # Move to '('
        if idx < len_ and toks[idx] == '(':
            idx += 1  # Skip '('
            args = []
            while idx < len_ and toks[idx] != ')':
                # Use parse_val_unit to handle arithmetic within function arguments
                idx, arg = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                args.append(arg)
                if idx < len_ and toks[idx] == ',':
                    idx += 1
            assert idx < len_ and toks[idx] == ')', f"Expected ')' after {func_name} arguments, got {toks[idx]}"
            idx += 1
            val = {'func': func_name, 'args': args}
        else:
            # If no '(' follows, treat it as a column
            idx = start_idx  # Reset to start
            idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
            val = col_unit
    # Handle CASE statements
    elif toks[idx].lower() == 'case':
        case_expr = {'case': []}  # Structure: {'case': [(condition, value), ...], 'else': value}
        idx += 1  # Skip 'case'
        
        while idx < len_ and toks[idx].lower() != 'end':
            if toks[idx].lower() == 'when':
                idx += 1  # Skip 'when'
                idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                if idx < len_ and toks[idx] in WHERE_OPS:
                    op_id = WHERE_OPS.index(toks[idx])
                    idx += 1
                    val1 = None
                    if op_id == WHERE_OPS.index("in"):
                        assert toks[idx] == '(', f"Expected '(' after IN, got {toks[idx]}"
                        idx += 1
                        val_list = []
                        while idx < len_ and toks[idx] != ')':
                            idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                            val_list.append(val)
                            if idx < len_ and toks[idx] == ',':
                                idx += 1
                        assert idx < len_ and toks[idx] == ')', f"Expected ')' after IN list, got {toks[idx]}"
                        idx += 1
                        val1 = val_list
                    else:
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    condition = (op_id, val_unit, val1)
                else:
                    condition = val_unit
                
                assert idx < len_ and toks[idx].lower() == 'then', f"Expected 'THEN' after WHEN, got {toks[idx]}"
                idx += 1  # Skip 'then'
                idx, result_val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                case_expr['case'].append((condition, result_val))
            
            elif toks[idx].lower() == 'else':
                idx += 1  # Skip 'else'
                idx, else_val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                case_expr['else'] = else_val
            else:
                raise ValueError(f"Unexpected token in CASE: {toks[idx]}")
        
        assert idx < len_ and toks[idx].lower() == 'end', f"Expected 'END' to close CASE, got {toks[idx]}"
        idx += 1  # Skip 'end'
        val = case_expr
    elif toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif toks[idx] == 'null':
        val = "NULL"
        idx += 1
    elif "\"" in toks[idx]:
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except ValueError:
            idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
            val = col_unit

    if isBlock:
        assert idx < len_ and toks[idx] == ')', f"Expected ')' but got {toks[idx]}"
        idx += 1
    return idx, val

def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []
    print(f"Parsing condition at {idx}: {toks[idx:]}")

    while idx < len_:
        not_op = False
        if toks[idx] == "(":
            idx += 1
            if idx < len_ and toks[idx] == "select":
                # Parse subquery as a value within parentheses
                idx, subquery = parse_sql(toks, idx, tables_with_alias, schema)
                assert idx < len_ and toks[idx] == ")", f"Expected ')' after subquery but got {toks[idx]}"
                idx += 1  # Skip ')'

                # Handle NOT before the operator
                if idx < len_ and toks[idx] == "not":
                    not_op = True
                    idx += 1

                # Look for the operator before the subquery by backtracking
                prev_idx = start_idx
                while prev_idx < len_ and toks[prev_idx] not in WHERE_OPS:
                    prev_idx += 1
                if prev_idx < len_ and toks[prev_idx] in WHERE_OPS:
                    op_id = WHERE_OPS.index(toks[prev_idx])
                    idx_before_subquery = prev_idx + 1  # Position after operator
                    assert idx_before_subquery <= len_ and toks[idx_before_subquery] == "(", f"Expected '(' after operator but got {toks[idx_before_subquery]}"
                    # Parse the left-hand side val_unit (before the subquery)
                    _, val_unit = parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables)
                    val1 = subquery
                    val2 = None

                    if op_id == WHERE_OPS.index("between"):
                        assert idx < len_ and toks[idx] == "and", f"Expected 'and' but got {toks[idx]}"
                        idx += 1
                        idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("in"):
                        assert idx < len_ and toks[idx] == "(", f"Expected '(' but got {toks[idx]}"
                        idx += 1
                        if toks[idx] == "select":
                            idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                        else:
                            val_list = []
                            while idx < len_ and toks[idx] != ")":
                                idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                                val_list.append(val)
                                if toks[idx] == ",":
                                    idx += 1
                            assert idx < len_ and toks[idx] == ")", f"Expected ')' but got {toks[idx]}"
                            idx += 1
                            val1 = val_list
                    elif op_id == WHERE_OPS.index("like"):
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("is"):
                        if idx < len_ and toks[idx] == "not":
                            not_op = True
                            idx += 1
                        assert idx < len_ and toks[idx] == "null", f"Expected 'null' but got {toks[idx]}"
                        val1 = "NULL"
                        idx += 1

                    conds.append((not_op, op_id, val_unit, val1, val2))
                else:
                    raise ValueError("Subquery in parentheses must be preceded by a comparison operator")
            else:
                # Parse regular condition inside parentheses
                idx, sub_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                assert idx < len_ and toks[idx] == ")", f"Expected ')' but got {toks[idx]}"
                idx += 1  # Skip ')'

                if idx < len_ and toks[idx] == "not":
                    not_op = True
                    idx += 1

                if idx < len_ and toks[idx] in WHERE_OPS:
                    op_id = WHERE_OPS.index(toks[idx])
                    idx += 1
                    val1 = val2 = None

                    if op_id == WHERE_OPS.index("between"):
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                        assert toks[idx] == "and", f"Expected 'and' but got {toks[idx]}"
                        idx += 1
                        idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("in"):
                        assert toks[idx] == "(", f"Expected '(' but got {toks[idx]}"
                        idx += 1
                        if toks[idx] == "select":
                            idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                        else:
                            val_list = []
                            while idx < len_ and toks[idx] != ")":
                                idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                                val_list.append(val)
                                if toks[idx] == ",":
                                    idx += 1
                            assert toks[idx] == ")", f"Expected ')' but got {toks[idx]}"
                            idx += 1
                            val1 = val_list
                    elif op_id == WHERE_OPS.index("like"):
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("is"):
                        if idx < len_ and toks[idx] == "not":
                            not_op = True
                            idx += 1
                        assert idx < len_ and toks[idx] == "null", f"Expected 'null' but got {toks[idx]}"
                        val1 = "NULL"
                        idx += 1
                    else:
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)

                    if len(sub_conds) == 1 and isinstance(sub_conds[0], tuple):
                        val_unit = sub_conds[0][2]
                        conds.append((not_op, op_id, val_unit, val1, val2))
                    else:
                        raise ValueError("Parenthesized condition must resolve to a single val_unit for comparison")
                else:
                    conds.extend(sub_conds)
        else:
            # Check for EXISTS or NOT EXISTS
            if idx < len_ and toks[idx] == "not":
                not_op = True
                idx += 1
            if idx < len_ and toks[idx] == "exists":
                idx += 1
                assert toks[idx] == "(", f"Expected '(' after EXISTS but got {toks[idx]}"
                idx += 1
                assert toks[idx] == "select", f"Expected subquery after EXISTS but got {toks[idx]}"
                idx, subquery = parse_sql(toks, idx, tables_with_alias, schema)
                assert toks[idx] == ")", f"Expected ')' after subquery but got {toks[idx]}"
                idx += 1
                op_id = WHERE_OPS.index("exists")
                conds.append((not_op, op_id, None, subquery, None))
            else:
                # Regular condition parsing
                idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
                if idx < len_ and toks[idx] == "not":
                    not_op = True
                    idx += 1

                if idx < len_ and toks[idx] in WHERE_OPS:
                    op_id = WHERE_OPS.index(toks[idx])
                    idx += 1
                    val1 = val2 = None

                    if op_id == WHERE_OPS.index("between"):
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                        assert toks[idx] == "and", f"Expected 'and' but got {toks[idx]}"
                        idx += 1
                        idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("in"):
                        assert toks[idx] == "(", f"Expected '(' but got {toks[idx]}"
                        idx += 1
                        if toks[idx] == "select":
                            idx, val1 = parse_sql(toks, idx, tables_with_alias, schema)
                        else:
                            val_list = []
                            while idx < len_ and toks[idx] != ")":
                                idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                                val_list.append(val)
                                if toks[idx] == ",":
                                    idx += 1
                            assert toks[idx] == ")", f"Expected ')' but got {toks[idx]}"
                            idx += 1
                            val1 = val_list
                    elif op_id == WHERE_OPS.index("like"):
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                    elif op_id == WHERE_OPS.index("is"):
                        if idx < len_ and toks[idx] == "not":
                            not_op = True
                            idx += 1
                        assert idx < len_ and toks[idx] == "null", f"Expected 'null' but got {toks[idx]}"
                        val1 = "NULL"
                        idx += 1
                    else:
                        idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)

                    conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1

        # Stop if we hit a join keyword (including 'left'), clause keyword, or closing parenthesis/semicolon
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS or toks[idx] == "left"):
            break

    print(f"Conditions: {conds}")
    return idx, conds

def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []
    aliases = {}

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] != ';':
        agg_id = AGG_OPS.index('none')
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        if toks[idx].endswith('.*'):  # Handle alias.* case
            idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
            val_unit = (UNIT_OPS.index('none'), (agg_id, col_id, False), None)
            val_units.append((agg_id, val_unit))
        else:
            idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx].lower() == 'as':
            idx += 1
            alias_name = toks[idx]
            aliases[alias_name] = val_units[-1]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1
        else:
            break

    return idx, (isDistinct, val_units), aliases

def parse_from(toks, start_idx, tables_with_alias, schema):
    assert 'from' in toks[start_idx:], "'from' not found"
    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    if idx < len_ and toks[idx] == '(':
        idx += 1  # Skip '('
        subquery_start = idx
        if toks[idx] == 'select':
            # Parse the subquery
            idx, sql = parse_sql(toks, subquery_start, tables_with_alias, schema)
            assert idx < len_ and toks[idx] == ')', f"Expected ')' after subquery, got {toks[idx]}"
            subquery_end = idx
            idx += 1  # Skip ')'
            alias = None
            if idx < len_ and toks[idx] == 'as':
                idx += 1
                alias = toks[idx]
                idx += 1
            elif (idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in JOIN_KEYWORDS and 
                  toks[idx] != ')' and toks[idx] != ';'):
                alias = toks[idx]
                idx += 1
            else:
                alias = f"subquery_{len(table_units)}"
            tables_with_alias[alias] = alias
            default_tables.append(alias)
            table_units.append((TABLE_TYPE['sql'], sql))

            # Correctly determine subquery output columns using aliases from SELECT
            sub_default_tables = [alias]
            sub_idx, sub_select, subquery_aliases = parse_select(toks, subquery_start, tables_with_alias, schema, default_tables=sub_default_tables)
            subquery_cols = []
            is_distinct, val_units = sub_select
            for i, (agg_id, val_unit) in enumerate(val_units):
                # Check if this val_unit has an alias in subquery_aliases
                alias_name = next((name for name, val in subquery_aliases.items() if val == (agg_id, val_unit)), None)
                if alias_name:
                    subquery_cols.append(alias_name)
                else:
                    # Fallback to column name if no alias is provided
                    col_id = val_unit[1][1]  # col_id from val_unit
                    col_name = None
                    for k, v in schema.idMap.items():
                        if v == col_id:
                            col_name = k.split('.')[-1]
                            break
                    if not col_name:
                        col_name = f'col{i}'
                    subquery_cols.append(col_name)
            schema.add_subquery_schema(alias, subquery_cols)
            print(f"Subquery schema for {alias}: {subquery_cols}")
        else:
            raise ValueError(f"Expected 'SELECT' after '(' in FROM clause, got {toks[idx]}")
    else:
        # Parse regular table
        idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
        table_units.append((TABLE_TYPE['table_unit'], table_unit))
        if (idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in JOIN_KEYWORDS and 
            toks[idx] != ')' and toks[idx] != ';'):
            alias = toks[idx]
            tables_with_alias[alias] = table_name
            default_tables.append(alias)
            idx += 1
        else:
            default_tables.append(table_name)

        # Handle JOIN clauses
        while idx < len_ and toks[idx] == 'join':
            idx += 1
            if idx < len_ and toks[idx] == '(':
                idx += 1
                subquery_start = idx
                if toks[idx] == 'select':
                    # Parse subquery in JOIN
                    idx, sql = parse_sql(toks, subquery_start, tables_with_alias, schema)
                    assert idx < len_ and toks[idx] == ')', f"Expected ')' after subquery, got {toks[idx]}"
                    subquery_end = idx
                    idx += 1
                    alias = None
                    if idx < len_ and toks[idx] == 'as':
                        idx += 1
                        alias = toks[idx]
                        idx += 1
                    elif (idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in JOIN_KEYWORDS and 
                          toks[idx] != ')' and toks[idx] != ';'):
                        alias = toks[idx]
                        idx += 1
                    else:
                        alias = f"subquery_{len(table_units)}"
                    tables_with_alias[alias] = alias
                    default_tables.append(alias)
                    table_units.append((TABLE_TYPE['sql'], sql))

                    # Determine subquery output columns
                    sub_default_tables = [alias]
                    sub_idx, sub_select, subquery_aliases = parse_select(toks, subquery_start, tables_with_alias, schema, default_tables=sub_default_tables)
                    subquery_cols = []
                    is_distinct, val_units = sub_select
                    for i, (agg_id, val_unit) in enumerate(val_units):
                        alias_name = next((name for name, val in subquery_aliases.items() if val == (agg_id, val_unit)), None)
                        if alias_name:
                            subquery_cols.append(alias_name)
                        else:
                            col_id = val_unit[1][1]
                            col_name = None
                            for k, v in schema.idMap.items():
                                if v == col_id:
                                    col_name = k.split('.')[-1]
                                    break
                            if not col_name:
                                col_name = f'col{i}'
                            subquery_cols.append(col_name)
                    schema.add_subquery_schema(alias, subquery_cols)
                    print(f"Subquery schema for {alias}: {subquery_cols}")
                else:
                    raise ValueError(f"Expected 'SELECT' after '(' in JOIN clause, got {toks[idx]}")
            else:
                # Parse regular table in JOIN
                idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
                table_units.append((TABLE_TYPE['table_unit'], table_unit))
                if (idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in JOIN_KEYWORDS and 
                    toks[idx] != ')' and toks[idx] != ';'):
                    alias = toks[idx]
                    tables_with_alias[alias] = table_name
                    default_tables.append(alias)
                    idx += 1
                else:
                    default_tables.append(table_name)

            # Parse ON or USING clause
            if idx < len_ and toks[idx] == "on":
                idx += 1
                idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
                if len(conds) > 0:
                    conds.append('and')
                conds.extend(this_conds)
            elif idx < len_ and toks[idx] == "using":
                idx += 1
                assert toks[idx] == '(', f"Expected '(' after USING, got {toks[idx]}"
                idx += 1
                col_name = toks[idx].lower()
                idx += 1
                assert toks[idx] == ')', f"Expected ')' after USING column, got {toks[idx]}"
                idx += 1

                table1 = table_units[-2][1]
                table2 = table_units[-1][1]
                table1_name = table1.split('__')[1]
                table2_name = table2.split('__')[1]
                col_id1 = schema.idMap[f"{table1_name}.{col_name}"]
                col_id2 = schema.idMap[f"{table2_name}.{col_name}"]
                val_unit = (UNIT_OPS.index('none'), (AGG_OPS.index('none'), col_id1, False), None)
                cond_unit = (False, WHERE_OPS.index('='), val_unit, (AGG_OPS.index('none'), col_id2, False), None)
                if len(conds) > 0:
                    conds.append('and')
                conds.append(cond_unit)

    if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        return idx, table_units, conds, default_tables

    return idx, table_units, conds, default_tables

def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    if idx >= len_ or toks[idx] != 'where':
        return idx, []
    idx += 1
    try:
        idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    except Exception as e:
        print(f"❌ Error in parse_where: {e}")
        raise
    return idx, conds

def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []  # Change to store val_units instead of col_units
    if idx >= len_ or toks[idx] != 'group':
        return idx, val_units
    idx += 1
    assert toks[idx] == 'by', f"Expected 'by' after 'group'"
    idx += 1
    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1
        else:
            break
    return idx, val_units

def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables, select_aliases):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'
    if idx >= len_ or toks[idx] != 'order':
        return idx, ('asc', [])
    idx += 1
    assert toks[idx] == 'by', f"Expected 'by' after 'order'"
    idx += 1
    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in (")", ";"):
        if toks[idx] in select_aliases:
            agg_id, val_unit = select_aliases[toks[idx]]
            # If there's an aggregation, wrap it into col_unit1, keeping val_unit as (unit_op, col_unit1, col_unit2)
            if agg_id != AGG_OPS.index("none"):
                # Extract the base col_unit and wrap it with the aggregation
                unit_op, base_col_unit, col_unit2 = val_unit
                col_unit1 = (agg_id, base_col_unit[1], base_col_unit[2])  # (agg_id, col_id, isDistinct)
                val_units.append((unit_op, col_unit1, col_unit2))
            else:
                val_units.append(val_unit)  # No aggregation, use as-is
            idx += 1
        else:
            idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1
        else:
            break
    return idx, (order_type, val_units)

def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    if idx >= len_ or toks[idx] != 'having':
        return idx, []
    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds

def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)
    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])
    return idx, None

def parse_sql(toks, start_idx, tables_with_alias, schema):
    try:
        isBlock = False
        len_ = len(toks)
        idx = start_idx
        sql = {
            'select': (False, []),
            'from': {'table_units': [], 'conds': []},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'union': None,
            'except': None
        }
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        from_start_idx = idx
        while from_start_idx < len_ and toks[from_start_idx].lower() != 'from':
            from_start_idx += 1
        if from_start_idx < len_:
            from_end_idx, table_units, conds, default_tables = parse_from(toks, from_start_idx, tables_with_alias, schema)
            sql['from'] = {'table_units': table_units, 'conds': conds}
        else:
            from_end_idx = len_
            default_tables = list(schema.schema.keys())

        select_idx = idx
        while select_idx < len_ and toks[select_idx].lower() != 'select':
            select_idx += 1
        if select_idx < len_:
            next_idx, select_col_units, select_aliases = parse_select(toks, select_idx, tables_with_alias, schema, default_tables)
            sql['select'] = select_col_units
        else:
            next_idx = idx
            select_aliases = {}

        idx = from_end_idx
        idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
        sql['where'] = where_conds
        idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
        sql['groupBy'] = group_col_units
        idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
        sql['having'] = having_conds
        idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables, select_aliases)
        sql['orderBy'] = order_col_units
        idx, limit_val = parse_limit(toks, idx)
        sql['limit'] = limit_val

        idx = skip_semicolon(toks, idx)
        if isBlock:
            if idx >= len_ or toks[idx] != ')':
                raise Exception("Missing closing parenthesis in SQL block")
            idx += 1
        idx = skip_semicolon(toks, idx)

        if idx < len_ and toks[idx] in SQL_OPS:
            sql_op = toks[idx]
            idx += 1
            idx, iue_sql = parse_sql(toks, idx, tables_with_alias, schema)
            sql[sql_op] = iue_sql

        def ensure_full_sql_structure(sql_dict):
            required_keys = {
                'select': (False, []),
                'from': {'table_units': [], 'conds': []},
                'where': [],
                'groupBy': [],
                'having': [],
                'orderBy': [],
                'limit': None,
                'intersect': None,
                'union': None,
                'except': None
            }
            for key, default in required_keys.items():
                if key not in sql_dict:
                    sql_dict[key] = default
            for key in ['intersect', 'union', 'except']:
                if sql_dict[key] is not None:
                    ensure_full_sql_structure(sql_dict[key])
            for cond in sql_dict['where']:
                if isinstance(cond, tuple) and len(cond) == 5:
                    if isinstance(cond[3], dict):
                        ensure_full_sql_structure(cond[3])
                    if isinstance(cond[4], dict):
                        ensure_full_sql_structure(cond[4])
            for cond in sql_dict['from']['conds']:
                if isinstance(cond, tuple) and len(cond) == 5:
                    if isinstance(cond[3], dict):
                        ensure_full_sql_structure(cond[3])
                    if isinstance(cond[4], dict):
                        ensure_full_sql_structure(cond[4])
            for cond in sql_dict['having']:
                if isinstance(cond, tuple) and len(cond) == 5:
                    if isinstance(cond[3], dict):
                        ensure_full_sql_structure(cond[3])
                    if isinstance(cond[4], dict):
                        ensure_full_sql_structure(cond[4])
            return sql_dict

        return idx, ensure_full_sql_structure(sql)
    except Exception as e:
        print(f"❌ Error in parse_sql: {e}")
        raise

def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data

def get_sql(schema, query):
    print(f"🔍 Parsing SQL for query: {query}")
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    print(f"Tables with alias: {tables_with_alias}")
    try:
        _, sql = parse_sql(toks, 0, tables_with_alias, schema)
        print(sql)
        return sql
    except Exception as e:
        print(f"❌ Error in get_sql: {e}")
        raise

def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx

if __name__ == "__main__":
    import os
    db_dir = "data/spider/test_database"
    db = "e_commerce"
    db_path = os.path.join(db_dir, db, db + ".sqlite")
    schema = Schema(get_schema(db_path))
    p_str = """SELECT email_address, town_city, county FROM Customers WHERE gender_code = (   SELECT gender_code   FROM Orders   JOIN Customers USING(customer_id)   GROUP BY gender_code   ORDER BY COUNT(*) ASC   LIMIT 1 );"""
    try:
        p_sql = get_sql(schema, p_str)
        # print("Parsed SQL:", json.dumps(p_sql, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")